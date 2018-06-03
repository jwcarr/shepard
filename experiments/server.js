// ------------------------------------------------------------------
// Experiment parameters
// ------------------------------------------------------------------

// GENERAL PARAMETERS -----------------------------------------------

// Node.js port number (must match line 6 of client.js)
var port = 9000;

// Job ID - arbitrary identifier for a particular run of the experiment
var job_id = '000000';

// Set to iterated_learning or category_learning
var experiment_type = 'iterated_learning';

// Number of stimulus items
var stimulus_set_size = 64;

// Number of training items that will be randomly selected from the full
// stimulus set
var training_set_size = 32;

// Number of training blocks and mini-test frequency (must be a divisor of
// training_set_size defined above). This defines how many times the
// training process will cycle through the training set and also how often
// a mini-test will be inserted to ensure that every training item is mini-
// tested exactly once
var n_blocks = 4;

// Every participant gets the baseline pay + a mini-test bonus per correct
// mini-test item + a test bonus per correct test item (in US cents)
var pay = {'baseline':300, 'mini_test':2, 'test':2};

// Regex validator for checking valid user IDs (e.g., a CrowdFlower ID is
// 8 digits long)
var id_validator = /^\d{8}$/;

// Participants will be termiated from the experiment if they fail to
// respond for a significant period of time. (Time expressed in seconds)
var terminate_timeout = 300;

// Cookie identifier to be placed in all user's browsers, to allow for
// future identification and exclusion
var cookie = 'shepard';

// Category label sets (each participant is assigned one of these sets in
// shuffled order). Every label is unique, each set contains no repeated
// letters, and the labels are meaningless in as many languages as possible.
// The size of the label sets (and thus the number of categories) is fixed
// at four, since changing this number would require modification to the
// page design to accommodate more buttons
var label_sets = [['pov','reb','wud','zix'], ['buv','jef','pid','zox'], ['fod','jes','wix','zuv'], ['gex','juf','vib','wop']];

// What type of test should be done at stage 2: production, comprehension,
// or communication?
var test_type = 'production';

// ITERATED LEARNING PARAMETERS -------------------------------------

// Maximum number of generations in an iterated learning chain
var max_generations = 10;

// Minimum acceptable entropy of button clicks in order for transmission to
// take place (measured in bits). If a participant's button click entropy
// is lower than this, their data will not be iterated
var min_entropy = 1.75;

// CATEGORY LEARNING PARAMETERS -------------------------------------

// If no one else is available for communication in a reasonable timeframe
// place the user into the backup_test_type version of the experiment
var backup_test_type = 'comprehension';
var backup_max_wait_time = 120;

// List of condition names
var conditions = ['angl', 'size', 'both'];

// Number of participants required in each condition
var n_participants = {'angl':40, 'size':40, 'both':40};

// Category partitions used for each condition. This object defines which
// category ID (0-3) each stimulus ID (0-63) belongs to. The actual
// geometric properties of each stimulus are defined in client.js (line 9)
var partitions = {
  'angl' : {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:1,17:1,18:1,19:1,20:1,21:1,22:1,23:1,24:1,25:1,26:1,27:1,28:1,29:1,30:1,31:1,32:2,33:2,34:2,35:2,36:2,37:2,38:2,39:2,40:2,41:2,42:2,43:2,44:2,45:2,46:2,47:2,48:3,49:3,50:3,51:3,52:3,53:3,54:3,55:3,56:3,57:3,58:3,59:3,60:3,61:3,62:3,63:3},
  'size' : {0:0,1:0,2:1,3:1,4:2,5:2,6:3,7:3,8:0,9:0,10:1,11:1,12:2,13:2,14:3,15:3,16:0,17:0,18:1,19:1,20:2,21:2,22:3,23:3,24:0,25:0,26:1,27:1,28:2,29:2,30:3,31:3,32:0,33:0,34:1,35:1,36:2,37:2,38:3,39:3,40:0,41:0,42:1,43:1,44:2,45:2,46:3,47:3,48:0,49:0,50:1,51:1,52:2,53:2,54:3,55:3,56:0,57:0,58:1,59:1,60:2,61:2,62:3,63:3},
  'both' : {0:0,1:0,2:0,3:0,4:1,5:1,6:1,7:1,8:0,9:0,10:0,11:0,12:1,13:1,14:1,15:1,16:0,17:0,18:0,19:0,20:1,21:1,22:1,23:1,24:0,25:0,26:0,27:0,28:1,29:1,30:1,31:1,32:2,33:2,34:2,35:2,36:3,37:3,38:3,39:3,40:2,41:2,42:2,43:2,44:3,45:3,46:3,47:3,48:2,49:2,50:2,51:2,52:3,53:3,54:3,55:3,56:2,57:2,58:2,59:2,60:3,61:3,62:3,63:3}
};

// If true, each new participant is assigned to one of the conditions at
// random. If false, the conditions are filled in order, which is useful in
// the case of the communication test_type where you want to place
// concurrent participants in the same condition to aid successful pairing
var assign_conditions_randomly = true;

// ------------------------------------------------------------------
// Server setup and parameter validation
// ------------------------------------------------------------------

// Import the required packages for setting up sockets and database
var app = require('express')();
var http = require('http').Server(app);
var io = require('socket.io')(http);
var db = require('mongojs')('shepard', ['users', 'chains', 'systems']);
var communication_queue = {}, communication_ready = {};

if (training_set_size > stimulus_set_size)
  throw new Error('training_set_size should be less than or equal to stimulus_set_size');

if (training_set_size % n_blocks !== 0)
  throw new Error('n_blocks must be a divisor of training_set_size');

label_sets.forEach(function(label_set) {
  if (label_set.length !== 4)
    throw new Error('Each label set must contain four labels');
});

switch (experiment_type) {

  case 'category_learning':
    if (['production','comprehension','communication'].indexOf(test_type) === -1)
      throw new Error('Under category_learning, test_type must be set to: \'production\', \'comprehension\', or \'communication\'');
    conditions.forEach(function(condition) {
      if (n_participants[condition] === undefined)
        throw new Error('You must define the n_participants required for the \'' + condition + '\' condition');
      if (partitions[condition] === undefined)
        throw new Error('You must define the category partition used by the \'' + condition + '\' condition');
      for (var i=0; i<stimulus_set_size; i++) {
        if (partitions[condition][i] === undefined)
          throw new Error('Stimulus ' + i + ' in not defined in the \'' + condition + '\' partition');
      }
      db.users.count({'job_id':job_id, 'condition':condition, 'status':'finished'}, function(err, n_finished) {
        n_participants[condition] -= n_finished;
        communication_queue[condition] = [];
        console.log(n_participants[condition] + ' participants permitted in the \'' + condition + '\' condition');
      });
    });
    break;

  case 'iterated_learning':
    if (['production','communication'].indexOf(test_type) === -1)
      throw new Error('Under iterated learning, test_type must be set to: \'production\' or \'communication\'');
    break;

  default:
    throw new Error('experiment_type must be set to: \'category_learning\' or \'iterated_learning\'');

}

// Calculate number of trials required based on parameters above
var n_training_trials = training_set_size * (n_blocks + 1);
var n_test_trials = stimulus_set_size;
var n_total_trials = n_training_trials + n_test_trials;

// Grid of 2x2 quadrants from the 8x8 stimulus space (used to randomly
// select items from the space in such a way that each category is equally
// likely to be sampled)
var grid = [[0,1,8,9], [2,3,10,11], [4,5,12,13], [6,7,14,15], [16,17,24,25], [18,19,26,27], [20,21,28,29], [22,23,30,31], [32,33,40,41], [34,35,42,43], [36,37,44,45], [38,39,46,47], [48,49,56,57], [50,51,58,59], [52,53,60,61], [54,55,62,63]];

// ------------------------------------------------------------------
// Functions
// ------------------------------------------------------------------

// Basic range function
function range(end_exclusive) {
  for (var array=[], i=0; i<end_exclusive; i++) {
    array.push(i);
  }
  return array;
}

// Create array of nulls of given length
function empty(length) {
  for (var array=[], i=0; i<length; i++) {
    array.push(null)
  }
  return array;
}

// Return a random integer
function randint(end_exclusive) {
  return Math.floor(Math.random() * end_exclusive);
}

// Fisher-Yates shuffle (array passed by reference)
function shuffle(array) {
  var counter = array.length, temp, index;
  while (counter) {
    index = randint(counter--);
    temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

// Return UNIX timestamp
function getTime() {
  return Math.floor(new Date() / 1000);
}

// Select a label set and shuffle it
function selectLabels() {
  var labels = label_sets[randint(label_sets.length)].slice();
  shuffle(labels);
  return labels;
}

// Generate a completion code
function generateCompletionCode() {
  var code = 'JC', chars = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z'];
  for (var i=0; i<12; i++) {
    code += chars[randint(24)];
  }
  code += 'SH';
  return code;
}

function selectTrainingItems() {
  var training_items = [];
  // For each quad in the grid, choose two items that will be trained
  grid.forEach(function(quad) {
    var quad_copy = quad.slice();
    training_items.push(quad_copy.splice(randint(4),1)[0]);
    training_items.push(quad_copy.splice(randint(3),1)[0]);
  });
  return training_items;
}

function createTrainingSequence(training_items) {
  var sequence = [], candidates = [], comm_mini_test = 0;
  // For each block of training
  for (var i=0; i<n_blocks; i++) {
    shuffle(training_items);
    for (var j=0; j<training_set_size; j++) {
      // Every n_blocks number of items, pop out an item from the candidates array and add it
      // into the sequence array as a mini test trial (but not on the first trial in a block)
      if (j % n_blocks === 0 && j > 0) {
        if (test_type === 'production')
          sequence.push(['production_mini_test', candidates.splice(randint(candidates.length),1)[0]]);
        else if (test_type === 'comprehension')
          sequence.push(['comprehension_mini_test', candidates.splice(randint(candidates.length),1)[0]]);
        else if (test_type === 'communication') {
          if (comm_mini_test % 2 === 0)
            sequence.push(['production_mini_test', candidates.splice(randint(candidates.length),1)[0]]);
          else
            sequence.push(['comprehension_mini_test', candidates.splice(randint(candidates.length),1)[0]]);
          comm_mini_test++;
        }
      }
      // If this is the first block, add the training item to the candidates array
      if (i === 0)
        candidates.push(training_items[j]);
      // Add the training item to the sequence array as a training_trial
      sequence.push(['training_trial', training_items[j]]);
    }
    // Add in the final mini-test for this block
    if (test_type === 'production')
      sequence.push(['production_mini_test', candidates.splice(randint(candidates.length),1)[0]]);
    else if (test_type === 'comprehension')
      sequence.push(['comprehension_mini_test', candidates.splice(randint(candidates.length),1)[0]]);
    else if (test_type === 'communication') {
      if (comm_mini_test % 2 === 0)
        sequence.push(['production_mini_test', candidates.splice(randint(candidates.length),1)[0]]);
      else
        sequence.push(['comprehension_mini_test', candidates.splice(randint(candidates.length),1)[0]]);
      comm_mini_test++;
    }
  }
  return sequence;
}

// Generate a length-64 sequence of 0-63 for production
function createProductionSequence() {
  var sequence = range(stimulus_set_size);
  shuffle(sequence);
  return sequence;
}

// Generate a length-64 sequence of 0-3 for comprehension
function createComprehensionSequence() {
  var sequence = [];
  for (var i=0; i<4; i++) {
    for (var j=0; j<stimulus_set_size/4; j++) {
      sequence.push(i)
    }
  }
  shuffle(sequence);
  return sequence;
}

function defaultUser(user_id, client_id, ip_address) {
  var time = getTime();
  var new_user = {
    user_id: user_id,
    client_id: client_id,
    ip_address: ip_address,
    job_id: job_id,
    status: 'running',
    stage: 'training',
    experiment_type: experiment_type,
    test_type: test_type,
    training_bonus: 0,
    test_bonus: 0,
    trial_n: 0,
    words: selectLabels(),
    completion_code: generateCompletionCode(),
    creation_time: time,
    modified_time: time,
    comments: null
  };
  return new_user
}

function checkClickEntropy(button_clicks) {
  var n_clicks = button_clicks.length, buttons = [0, 0, 0, 0], summation = 0.0;
  for (var i=0; i<n_clicks; i++) {
    buttons[button_clicks[i]]++;
  }
  for (var i=0; i<4; i++) {
    var p = buttons[i] / n_clicks;
    if (p > 0)
      summation += p * Math.log2(p);
  }
  return -summation;
}

function checkCategoryCollapse(sequence) {
  var categories = [0, 0, 0, 0];
  for (var i=0; i<sequence.length; i++) {
    categories[sequence[i]]++;
  }
  if (categories[0] === 64 || categories[1] === 64 || categories[2] === 64 || categories[3] === 64)
    return true;
  return false;
}

function removeUser(user) {
  if (user.experiment_type === 'iterated_learning') {
    db.chains.update({ $and:[ {job_id:user.job_id}, {chain_id:user.chain_id} ] }, { $set:{running_user:null}, $push:{terminated_users:user.user_id} }, function(err, updated) {
      if (err || !updated)
        return console.log('Database error: Unable to remove user from chain.');
    });
  } else if (user.experiment_type === 'category_learning') {
    db.systems.update({ $and:[ {job_id:user.job_id}, {system:user.system} ] }, { $push:{terminated_users:user.user_id} }, function(err, updated) {
      if (err || !updated)
        return console.log('Database error: Unable to remove user from chain.');
    });
    n_participants[user.condition]++;
  }
  if (user.test_type === 'communication') {
    var queue_index = communication_queue[user.condition].indexOf(user.user_id);
    if (queue_index !== -1)
      communication_queue[user.condition].splice(queue_index, 1);
    if (user.partner_id) {
      db.users.findAndModify({ query:{ user_id:user.partner_id }, update:{ $set:{status:'terminated_by_partner'} } }, function(err, partner, last_err) {
        if (err || !partner)
          return console.log('Database error: Not able to update partner entry in database');
        var total_pay = pay.baseline + partner.training_bonus + partner.test_bonus;
        try {
          io.sockets.connected[partner.client_id].emit('partner_disconnect', { completion_code:partner.completion_code, baseline_pay:pay.baseline, training_bonus:partner.training_bonus, test_bonus:partner.test_bonus, total_pay:total_pay, cookie });
        } catch (err) {
          console.log('Could not notify ' + partner.user.user_id + ' about partner disconnect');
        }
      });
    }
  }
}

// Iterated learning functions --------------------------------------

if (experiment_type === 'iterated_learning') {

  var checkAvailability = function(cbAvailability) {
    db.chains.count({ $and:[ { job_id:job_id }, { running_user:null } ] }, function(err, n_available_chains) {
      if (err)
        return console.log('Database error: Unable to retrieve open chains.');
      return setImmediate(function() { cbAvailability(n_available_chains > 0); });
    });
  };

  var assignUser = function(user_id, cbSuccessfulChainAssignment, cbFailureToAssignChain) {
    db.chains.findAndModify({
      query  : { $and:[ { job_id:job_id }, { running_user:null } ] },
      update : { $set:{ running_user:user_id } },
      sort   : { chain_id: 1 }
    }, function(err, assigned_chain, last_err) {
      if (err || !assigned_chain)
        return setImmediate(function() { cbFailureToAssignChain(); });
      return setImmediate(function() { cbSuccessfulChainAssignment(assigned_chain); });
    });
  };

  var createNewUser = function(user_id, client_id, ip_address, cbSuccessfullyCreatedNewUser, cbFailureToCreateNewUser) {
    assignUser(user_id,
      function(assignation) { // cbSuccessfulChainAssignment
        var new_user = defaultUser(user_id, client_id, ip_address);
        new_user['chain_id'] = assignation.chain_id;
        new_user['generation'] = assignation.generations.length;
        new_user['partition'] = assignation.generations[new_user.generation-1].partition;
        new_user['training_sequence'] = createTrainingSequence(assignation.generations[new_user.generation-1].training_out);
        db.users.save(new_user, function(err, saved) {
          if (err || !saved)
            return setImmediate(function() { cbFailureToCreateNewUser(); });
          return setImmediate(function() { cbSuccessfullyCreatedNewUser(new_user); });
        });
      },
      function() { // cbFailureToAssignChain
        return setImmediate(function() { cbFailureToCreateNewUser(); });
      }
    );
  };

  var createNextGeneration = function(user) {
    if (checkClickEntropy(user.test_positions) < min_entropy) {
      db.chains.update({ job_id:job_id, chain_id:user.chain_id }, { $set:{ running_user:null }, $push:{ rejected_users:user.user_id } }, function(err, updated) {
        if (err || !updated)
          return console.log('Failed to reopen chain.');
      });
      return console.log('Could not produce new generation from user ' + user.user_id + '\'s data; entropy too low.');
    }
    if (user.test_bonus === (stimulus_set_size*pay.test)) {
      var running_status = false;
      console.log('Chain ' + user.chain_id + ' has fixated.');
    }
    else if (checkCategoryCollapse(user.test_responses)) {
      var running_status = false;
      console.log('Chain ' + user.chain_id + ' has collapsed.');
    }
    else if (user.generation >= max_generations) {
      var running_status = false;
      console.log('Chain ' + user.chain_id + ' has reached max generations.');
    }
    else
      var running_status = null;
    var partition = empty(stimulus_set_size), training_out = selectTrainingItems();
    for (var i=0; i<stimulus_set_size; i++) {
      partition[user.test_sequence[i]] = user.test_responses[i];
    }
    var generation = {user_id:user.user_id, partition, training_out};
    db.chains.update({ job_id:job_id, chain_id:user.chain_id }, { $set:{ running_user:running_status }, $push:{ finished_users:user.user_id, generations:generation } }, function(err, updated) {
      if (err || !updated)
        return console.log('Failed to create new generation.');
    });
  };

}

// Category learning functions --------------------------------------

if (experiment_type === 'category_learning') {

  var checkAvailability = function(cbAvailability) {
    db.systems.find({ available_slots:{$gt:0} }, function(err, available_systems) {
      if (err)
        return console.log('Database error: Unable to retrieve open systems.');
      setImmediate(function() { cbAvailability(available_systems); });
    });
  };

  // Find a condition that still requires participants, subtract 1 from its count,
  // and return the name of condition.
  var assignUser = function(cbSuccessfulSystemAssignment, cbFailureToAssignSystem) {
    checkAvailability(function(available_systems) {
      var n_available_systems = available_systems.length;
      if (n_available_systems > 0) {
        var assigned_system = available_systems[randint(n_available_systems)];
        db.systems.findAndModify({
          query  : { $and: [ {system_id:assigned_system.system_id}, {available_slots:{$gt:0}} ] },
          update : { $inc: { available_slots:-1 } }
        }, function(err, system, last_err) {
          if (err)
            return console.log('error');
          if (!system)
            assignUser(cbSuccessfulSystemAssignment, cbFailureToAssignSystem);
          else
            setImmediate(function() { cbSuccessfulSystemAssignment(assigned_system); });
        });
      } else
        setImmediate(function() { cbFailureToAssignSystem(); });
    });
  };

  var createNewUser = function(user_id, client_id, ip_address, cbSuccessfullyCreatedNewUser, cbFailureToCreateNewUser) {
    assignUser(user_id,
      function(assignation) { // cbSuccessfulSystemAssignment
        var new_user = defaultUser(user_id, client_id, ip_address);
        new_user['system'] = assignation.system_id;
        new_user['partition'] = partitions[assignation.system_id];
        new_user['training_sequence'] = createTrainingSequence(selectTrainingItems());
        db.users.save(new_user, function(err, saved) {
          if (err || !saved)
            return setImmediate(function() { cbFailureToCreateNewUser(); });
          return setImmediate(function() { cbSuccessfullyCreatedNewUser(new_user); });
        });
      },
      function() { // cbFailureToAssignSystem
        return setImmediate(function() { cbFailureToCreateNewUser(); });
      }
    );
  };

}

// Communication functions ------------------------------------------

if (test_type === 'communication') {

  var pairUsers = function(user1, user2, condition) {
    db.users.findOne({ user_id:user1 }, function(err, user) {
      if (err || !user)
        return console.log('Database error: Unable to access user.');
      var user1_client_id = user.client_id;
      try {
        io.sockets.connected[user1_client_id].emit('communication_line_test');
      } catch (err) {
        communication_queue[condition].unshift(user2);
        return console.log('Could not contact ' + user1 + '. Placing ' + user2 + ' at the front of the queue.');
      }
      db.users.findOne({ user_id:user2 }, function(err, user) {
        if (err || !user)
          return console.log('Database error: Unable to access user.');
        var user2_client_id = user.client_id;
        try {
          io.sockets.connected[user2_client_id].emit('communication_line_test');
        } catch (err) {
          communication_queue[condition].unshift(user1);
          return console.log('Could not contact ' + user2 + '. Placing ' + user1 + ' at the front of the queue.');
        }
        var test_sequence = createProductionSequence();
        db.users.findAndModify({ query:{ user_id:user1 }, update:{ $set:{test_sequence:test_sequence, trial_n:0, status:'running', stage:'communication', first:true, partner_id:user2} } }, function(err, user, last_err) {
          if (err)
            return console.log('Database error: Unable to update first paired user.');
          db.users.findAndModify({ query:{ user_id:user2 }, update:{ $set:{test_sequence:test_sequence, trial_n:0, status:'running', stage:'communication', first:false, partner_id:user1} } }, function(err, user, last_err) {
            if (err)
              return console.log('Database error: Unable to update second paired user.');
            communication_ready[user1] = false;
            communication_ready[user2] = false;
            try {
              io.sockets.connected[user1_client_id].emit('communication_paired');
              io.sockets.connected[user2_client_id].emit('communication_paired');
            } catch (err) {
              return console.log('Failed to pair users');
            }
          });
        });
      });
    });
  };

  var estimateWaitTime = function(user_id, condition, imaginary_insert, callback) {
    var num_in_queue = communication_queue[condition].length;
    if (imaginary_insert) {
      num_in_queue += 1;
      var last_in_queue = true;
    } else
      var last_in_queue = (user_id === communication_queue[condition][num_in_queue-1]);
    if (num_in_queue % 2 === 1 && last_in_queue) {
      db.users.find({ user_id:{$ne:user_id}, test_type:'communication', status:'running', stage:'training', condition:condition }, function(err, users) {
        if (err)
          return console.log('Database error: Unable to retrieve matching users.');
        var shortest_estimated_time = 3600;
        if (users)
          users.forEach(function(usr) {
            var remaining_trials = n_training_trials - usr.trial_n;
            var user_estimated_time = remaining_trials * ((getTime()-usr.creation_time)/(usr.trial_n+6)) + 30;
            if (user_estimated_time < shortest_estimated_time)
              shortest_estimated_time = user_estimated_time;
          });
        setImmediate(function() { callback(shortest_estimated_time); });
      });
    } else
      setImmediate(function() { callback(10); });
  };

}

// ------------------------------------------------------------------
// Client event handlers
// ------------------------------------------------------------------

io.sockets.on('connection', function(client) {

  // Check if there's space available on the experiment
  client.on('availability', function() { // cbAvailability
    checkAvailability(function(available) {
      client.emit('availability', { available });
    });
  });

  // Attempt to register a user onto the experiment
  client.on('register', function(payload) {
    if (!id_validator.test(payload.user_id))
      return client.emit('reject_id', { message:'Invalid user ID' });
    var ip_address = client.request.connection.remoteAddress;
    db.users.count({ $or: [ { user_id:payload.user_id }, { ip_address:ip_address } ] }, function(err, n_users) {
      if (err)
        return console.log('Database error: Unable to count users');
      if (n_users > 0)
        return client.emit('reject_id', { message:'Sorry, this task is no longer available', cookie });
      createNewUser(payload.user_id, client.id, ip_address,
        function(new_user) { // cbSuccessfullyCreatedNewUser
          client.emit('consent', { words:new_user.words, test_type:new_user.test_type, partition:new_user.partition, cookie });
          console.log(payload.user_id + ' has registered');
        },
        function() { // cbFailureToCreateNewUser
          client.emit('reject_id', { message:'Sorry, this task is currently busy. Please try again later. (error 1)' });
        }
      );
    });
  });

// Training ---------------------------------------------------------

  client.on('start_training', function(payload) {
    db.users.findOne({ user_id:payload.user_id }, function(err, user) {
      if (err || !user)
        return console.log('Database error: Unable to access user.');
      if (user.status !== 'running')
        return client.emit('timeout', { cookie });
      var stim_n = user.training_sequence[0][1];
      var cat_n = user.partition[stim_n];
      var progress = 1 / n_training_trials;
      client.emit('training_trial', { stim_n, cat_n, progress, money:0 });
    });
  });

  client.on('training_response', function(payload) {
    db.users.findOne({ user_id:payload.user_id }, function(err, user) {
      if (err || !user)
        return console.log('Database error: Unable to access user.');
      if (user.status !== 'running')
        return client.emit('timeout', { cookie });
      var prev_trial = user.training_sequence[user.trial_n];
      var prev_cat_n = user.partition[prev_trial[1]];
      var next_trial = user.training_sequence[user.trial_n+1];
      var progress = (user.trial_n+2) / n_training_trials;
      if (prev_trial[0] === 'production_mini_test') {
        if (payload.selection === prev_cat_n)
          user.training_bonus += pay.mini_test;
      } else if (prev_trial[0] === 'comprehension_mini_test') {
        var response_cat_n = user.partition[payload.selection];
        if (response_cat_n === prev_cat_n)
          user.training_bonus += pay.mini_test;
      }
      db.users.update({ user_id:user.user_id }, { $inc:{trial_n:1}, $push:{training_responses:payload.selection, training_positions:payload.button_id, training_reaction_times:payload.reaction_time}, $set:{training_bonus:user.training_bonus, modified_time:getTime()} }, function(err, updated) {
        if (err || !updated)
          return console.log('Database error: Unable to save response.');
      });
      if (next_trial) {
        var next_cat_n = user.partition[next_trial[1]];
        if (next_trial[0] === 'training_trial')
          return client.emit('training_trial', { stim_n:next_trial[1], cat_n:next_cat_n, progress, money:user.training_bonus });
        else
          return client.emit(next_trial[0], { stim_n:next_trial[1], cat_n:next_cat_n, progress, money:user.training_bonus, bonus:pay.mini_test });
      }
      switch (user.test_type) {
        case 'production':
          return client.emit('production_instructions');
        case 'comprehension':
          return client.emit('comprehension_instructions');
        case 'communication':
          estimateWaitTime(user.user_id, user.condition, true, function(estimated_time) {
            if (estimated_time < backup_max_wait_time)
              return client.emit('communication_instructions');
            switch (backup_test_type) {
              case 'production':
                // update user.test_type in DB
                return client.emit('production_instructions');
              case 'comprehension':
                // update user.test_type in DB
                return client.emit('comprehension_instructions');
            }
          });
      }
    });
  });

// Production -------------------------------------------------------

  client.on('production_start', function(payload) {
    var test_sequence = createProductionSequence();
    db.users.findAndModify({ query:{ user_id:payload.user_id }, update:{ $set:{test_sequence:test_sequence, trial_n:0, stage:'production'} } }, function(err, user, last_err) {
      if (err || !user)
        return console.log('Database error: Unable to access user.');
      if (user.status !== 'running')
        return client.emit('timeout', { cookie });
      client.emit('production_trial', { stim_n:test_sequence[0], progress:1/n_test_trials, bonus:pay.test });
    });
  });

  client.on('production_response', function(payload) {
    db.users.findOne({ user_id:payload.user_id }, function(err, user) {
      if (err || !user)
        return console.log('Database error: Unable to access user.');
      if (user.status !== 'running')
        return client.emit('timeout', { cookie });
      var prev_trial = user.test_sequence[user.trial_n];
      var prev_cat_n = user.partition[prev_trial];
      if (payload.selection === prev_cat_n)
        user.test_bonus += pay.test;
      if (user.trial_n+1 === n_test_trials) {
        var total_pay = pay.baseline + user.training_bonus + user.test_bonus;
        client.emit('end', { completion_code:user.completion_code, baseline_pay:pay.baseline, training_bonus:user.training_bonus, test_bonus:user.test_bonus, total_pay:total_pay, cookie });
        db.users.findAndModify({ query: { user_id:user.user_id }, update: { $push:{test_responses:payload.selection, test_positions:payload.button_id, test_reaction_times:payload.reaction_time}, $set:{test_bonus:user.test_bonus, total_pay:total_pay, modified_time:getTime(), status:'finished'} }, new:true }, function(err, updated_user, last_err) {
          if (err || !updated_user)
            return console.log('Database error');
          if (experiment_type === 'iterated_learning')
            setImmediate(function() { createNextGeneration(updated_user); });
        });
        console.log(user.user_id + ' has finished');
      } else {
        var next_trial = user.test_sequence[user.trial_n+1];
        var progress = (user.trial_n+2) / n_test_trials;
        client.emit('production_trial', { stim_n:next_trial, progress, bonus:pay.test });
        db.users.update({ user_id:user.user_id }, { $inc:{trial_n:1}, $push:{test_responses:payload.selection, test_positions:payload.button_id, test_reaction_times:payload.reaction_time}, $set:{test_bonus:user.test_bonus, modified_time:getTime()} }, function(err, updated) {
          if (err || !updated)
            return console.log('Database error: Unable to save response.');
        });
      }
    });
  });

// Comprehension ----------------------------------------------------

  client.on('comprehension_start', function(payload) {
    var test_sequence = createComprehensionSequence();
    db.users.findAndModify({ query:{ user_id:payload.user_id }, update:{ $set:{test_sequence:test_sequence, trial_n:0, stage:'comprehension'} } }, function(err, user, last_err) {
      if (err || !user)
        return console.log('Database error: Unable to access user.');
      if (user.status !== 'running')
        return client.emit('timeout', { cookie });
      client.emit('comprehension_trial', { cat_n:test_sequence[0], progress:1/n_test_trials, bonus:pay.test });
    });
  });

  client.on('comprehension_response', function(payload) {
    db.users.findOne({ user_id:payload.user_id }, function(err, user) {
      if (err || !user)
        return console.log('Database error: Unable to access user.');
      if (user.status !== 'running')
        return client.emit('timeout', { cookie });
      var prev_trial = user.test_sequence[user.trial_n];
      var response_cat_n = user.partition[payload.selection];
      if (prev_trial === response_cat_n)
        user.test_bonus += pay.test;
      if (user.trial_n+1 === n_test_trials) {
        var total_pay = pay.baseline + user.training_bonus + user.test_bonus;
        db.users.update({ user_id:user.user_id }, { $inc:{trial_n:1}, $push:{test_responses:payload.selection, test_positions:payload.button_id, test_reaction_times:payload.reaction_time}, $set:{test_bonus:user.test_bonus, total_pay:total_pay, modified_time:getTime(), status:'finished'} }, function(err, updated) {
          if (err || !updated)
            return console.log('Database error: Unable to save response.');
        });
        client.emit('end', { completion_code:user.completion_code, baseline_pay:pay.baseline, training_bonus:user.training_bonus, test_bonus:user.test_bonus, total_pay:total_pay, cookie });
        console.log(user.user_id + ' has finished');
      } else {
        var next_trial = user.test_sequence[user.trial_n+1];
        var progress = (user.trial_n+2) / n_test_trials;
        db.users.update({ user_id:user.user_id }, { $inc:{trial_n:1}, $push:{test_responses:payload.selection, test_positions:payload.button_id, test_reaction_times:payload.reaction_time}, $set:{test_bonus:user.test_bonus, modified_time:getTime()} }, function(err, updated) {
          if (err || !updated)
            return console.log('Database error: Unable to save response.');
        });
        client.emit('comprehension_trial', { cat_n:next_trial, progress, bonus:pay.test });
      }
    });
  });

// Communication ----------------------------------------------------

  client.on('communication_start', function(payload) {
    db.users.findAndModify({ query:{ user_id:payload.user_id }, update:{ $set:{status:'waiting'} } }, function(err, user, last_err) {
      if (err || !user)
        return console.log('Database error: Unable to access user.');
      communication_queue[user.condition].push(user.user_id);
      estimateWaitTime(user.user_id, user.condition, false, function(wait_time) {
        client.emit('communication_wait', { wait_time });
      });
    });
  });

  client.on('communication_estimate_wait', function(payload) {
    db.users.findOne({ user_id:payload.user_id }, function(err, user) {
      if (err || !user)
        return console.log('Database error: Unable to access user.');
      estimateWaitTime(user.user_id, user.condition, false, function(wait_time) {
        client.emit('communication_wait', { wait_time });
      });
    });
  });

  client.on('communication_next', function(payload) {
    db.users.findOne({ user_id:payload.user_id }, function(err, user) {
      if (err || !user)
        return console.log('Database error: Unable to access user.');
      communication_ready[payload.user_id] = true;
      if (communication_ready[user.partner_id]) {
        communication_ready[user.user_id] = false;
        communication_ready[user.partner_id] = false;
        db.users.findOne({ user_id:user.partner_id }, function(err, partner) {
          if (err || !partner)
            return console.log('Database error: Unable to access partner.');
          if (user.trial_n % 2 == user.first) {
            try {
              io.sockets.connected[partner.client_id].emit('communication_trial', { stim_n:user.test_sequence[user.trial_n], progress:(user.trial_n+1)/n_test_trials, bonus:pay.test, director:true });
              client.emit('communication_trial', { stim_n:user.test_sequence[user.trial_n], progress:(user.trial_n+1)/n_test_trials, bonus:pay.test, director:false });
            } catch (err) {
              return console.log('Could not initialize a new communication session');
            }
            db.users.update({ user_id:user.user_id }, { $inc:{trial_n:1}, $set:{modified_time:getTime(), status:'waiting'} }, function(err, updated) {
              if (err || !updated)
                return console.log('Database error: Unable to save response.');
            });
            db.users.update({ user_id:partner.user_id }, { $inc:{trial_n:1}, $set:{modified_time:getTime(), status:'running'} }, function(err, updated) {
              if (err || !updated)
                return console.log('Database error: Unable to save response.');
            });
          } else {
            try {
              io.sockets.connected[partner.client_id].emit('communication_trial', { stim_n:user.test_sequence[user.trial_n], progress:(user.trial_n+1)/n_test_trials, bonus:pay.test, director:false });
              client.emit('communication_trial', { stim_n:user.test_sequence[user.trial_n], progress:(user.trial_n+1)/n_test_trials, bonus:pay.test, director:true });
            } catch (err) {
              return console.log('Could not initialize a new communication session');
            }
            db.users.update({ user_id:user.user_id }, { $inc:{trial_n:1}, $set:{modified_time:getTime(), status:'running'} }, function(err, updated) {
              if (err || !updated)
                return console.log('Database error: Unable to save response.');
            });
            db.users.update({ user_id:partner.user_id }, { $inc:{trial_n:1}, $set:{modified_time:getTime(), status:'waiting'} }, function(err, updated) {
              if (err || !updated)
                return console.log('Database error: Unable to save response.');
            });
          }
        });
      }
    });
  });

  client.on('communication_signal', function(payload) {
    db.users.findAndModify({ query:{ user_id:payload.user_id }, update:{ $push:{prod_responses:payload.selection, prod_positions:payload.button_id, prod_reaction_times:payload.reaction_time}, $set:{modified_time:getTime(), status:'waiting'} } }, function(err, user, last_err) {
      if (err || !user)
        return console.log('Database error: Unable to save comm response.');
      db.users.findAndModify({ query:{ user_id:user.partner_id }, update:{ $set:{modified_time:getTime(), status:'running'} } }, function(err, partner, last_err) {
        if (err || !partner)
          return console.log('Database error: Unable to set user status to running.');
        try {
          io.sockets.connected[partner.client_id].emit('communication_pass_signal', { signal:payload.selection });
        } catch (err) {
          return console.log('Could not pass signal to matcher');
        }
      });
    });
  });

  client.on('communication_feedback', function(payload) {
    db.users.findAndModify({ query:{ user_id:payload.user_id }, update:{ $push:{comp_responses:payload.selection, comp_positions:payload.button_id, comp_reaction_times:payload.reaction_time}, $set:{modified_time:getTime(), status:'waiting'} } }, function(err, user, last_err) {
      if (err || !user)
        return console.log('Database error: Unable to save comm feedback.');
      db.users.findAndModify({ query:{ user_id:user.partner_id }, update:{ $set:{modified_time:getTime(), status:'running'} } }, function(err, partner, last_err) {
        if (err || !partner)
          return console.log('Database error: Unable to set user status to running.');
        try {
          io.sockets.connected[partner.client_id].emit('communication_pass_feedback', { feedback:payload.selection });
        } catch (err) {
          return console.log('Could not pass feedback to director');
        }
      });
    });
  });

// Comments ---------------------------------------------------------

  client.on('send_comments', function(payload) {
    db.users.update({ user_id:payload.user_id }, { $set:{comments:payload.comments} }, function(err, updated) {
      if (err || !updated)
        return console.log('Database error: Unable to save comments.');
    });
  });

// Termination ------------------------------------------------------

  client.on('terminate', function(payload) {
    db.users.findAndModify({ query : { user_id:payload.user_id }, update:{ $set:{status:'terminated'} } }, function(err, user, last_err) {
      if (err || !user)
        return console.log('Database error: Unable to change user status to terminated.');
      client.emit('early_exit', { completion_code:user.completion_code, cookie });
      console.log(user.user_id + ' has terminated');
      setImmediate(function(){ removeUser(user); });
    });
  });

  client.on('disconnect', function() { // For communication, status should be 'running' or 'waiting'
    db.users.findAndModify({ query : { client_id:client.id, $or:[{status:'running'}, {status:'waiting'}] }, update:{ $set:{status:'terminated'} } }, function(err, user, last_err) {
      if (err)
        return console.log('Database error: Unable to change user status to terminated.');
      if (user) {
        if (user.status === 'running') {
          console.log(user.user_id + ' has disconnected');
          setImmediate(function(){ removeUser(user); });
        }
      }
    });
  });

});

// ------------------------------------------------------------------
// Periodic processes (running at prime intervals to reduce load)
// ------------------------------------------------------------------

// Periodically check the communication_queue and attempt to pair participants up
if (test_type === 'communication') {
  setInterval(function() {
    conditions.forEach(function(condition) {
      if (communication_queue[condition].length > 1) {
        var user1 = communication_queue[condition].shift();
        var user2 = communication_queue[condition].shift();
        pairUsers(user1, user2, condition);
      }
    });
  }, 7369); // Some prime number around 7 seconds
}

// Periodically find users who are inactive and terminate them
setInterval(function() {
  var cut_off_timeout = getTime() - terminate_timeout;
  db.users.find({ $and: [ { status:'running' }, { modified_time:{$lt:cut_off_timeout} } ] }, function(err, users) {
    if (err || !users)
      return console.log('Database error: Unable to search users.');
    users.forEach(function(user) {
      try {
        io.sockets.connected[user.client_id].emit('timeout', { cookie });
        console.log(user.user_id + ' has been auto-terminated');
      } catch (err) {
        console.log('Could not notify user of auto-termination');
      }
      db.users.update({ user_id:user.user_id }, { $set:{status:'terminated'} }, function(err, updated) {
        if (err || !updated)
          return console.log('Database error: Unable to mark user as terminated.');
      });
      setImmediate(function(){ removeUser(user); });
    });
  });
}, 13729); // Some prime number around 13 seconds

// ------------------------------------------------------------------
// Listen out for some clients
// ------------------------------------------------------------------

setTimeout(function() {
  http.listen(port);
  console.log('Listening on port ' + port);
}, 1000);
