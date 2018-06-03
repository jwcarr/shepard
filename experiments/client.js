// ------------------------------------------------------------------
// Experiment parameters
// ------------------------------------------------------------------

// Node.js port number (must match line 6 of server.js)
var port = 9000;

// Angle (in radians) and radius (in pixels) for each of the 64 Shepard circles
var shepard_circles = [[2.5656,25],[2.5656,50],[2.5656,75],[2.5656,100],[2.5656,125],[2.5656,150],[2.5656,175],[2.5656,200],[3.0144,25],[3.0144,50],[3.0144,75],[3.0144,100],[3.0144,125],[3.0144,150],[3.0144,175],[3.0144,200],[3.4632,25],[3.4632,50],[3.4632,75],[3.4632,100],[3.4632,125],[3.4632,150],[3.4632,175],[3.4632,200],[3.912,25],[3.912,50],[3.912,75],[3.912,100],[3.912,125],[3.912,150],[3.912,175],[3.912,200],[4.3608,25],[4.3608,50],[4.3608,75],[4.3608,100],[4.3608,125],[4.3608,150],[4.3608,175],[4.3608,200],[4.8096,25],[4.8096,50],[4.8096,75],[4.8096,100],[4.8096,125],[4.8096,150],[4.8096,175],[4.8096,200],[5.2583,25],[5.2583,50],[5.2583,75],[5.2583,100],[5.2583,125],[5.2583,150],[5.2583,175],[5.2583,200],[5.7072,25],[5.7072,50],[5.7072,75],[5.7072,100],[5.7072,125],[5.7072,150],[5.7072,175],[5.7072,200]];

// Display a warning message to notify the participant that they will soon be
// terminated if they remain inactive. (Time expressed in seconds)
var timeout_warning = 150;

// Regex validator for valid user IDs (e.g., CrowdFlower ID is 8 digits)
var id_validator = /^\d{8}$/;

// ------------------------------------------------------------------
// Declare globals and connect to the server
// ------------------------------------------------------------------

var user_id, words, partition, timeout_timer, wait_timer, wait_estimation_timer;
var server = io.connect(location.host + ':' + port);

// ------------------------------------------------------------------
// Initialize the canvases
// ------------------------------------------------------------------

var canvas_middle = document.getElementById('canvas_middle').getContext('2d');
canvas_middle.lineJoin = 'round', canvas_middle.lineCap = 'round', canvas_middle.lineWidth = 2;

var canvas_left = document.getElementById('canvas_left').getContext('2d');
canvas_left.lineJoin = 'round', canvas_left.lineCap = 'round', canvas_left.lineWidth = 2;

var canvas_right = document.getElementById('canvas_right').getContext('2d');
canvas_right.lineJoin = 'round', canvas_right.lineCap = 'round', canvas_right.lineWidth = 2;

var comprehension_canvases = [];
for (var i=0; i<64; i++) {
  var stim_button = document.getElementById('stim_button_' + i).getContext('2d');
  stim_button.lineJoin = 'round', stim_button.lineCap = 'round', stim_button.lineWidth = 1;
  comprehension_canvases.push(stim_button);
}

// ------------------------------------------------------------------
// General functions
// ------------------------------------------------------------------

function shuffle(array) {
  var counter = array.length, temp, index;
  while (counter) {
    index = Math.floor(Math.random() * counter--);
    temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

// Timeouts ---------------------------------------------------------

function startTimeout() {
  timeout_timer = setTimeout(function() {
    displayAlert('Warning: Your session will be terminated in 2 minutes. Click Continue to carry on with the task. If you do not wish to complete the task, please click Exit; you will only receive 1&cent; if you Exit now.',
      function() {
        startTimeout();
      },
      function() {
        server.emit('terminate', { user_id });
        $('#experiment').hide();
        $('#consent').hide();
        $('#training_instructions').hide();
        $('#production_instructions').hide();
        $('#comprehension_instructions').hide();
        $('#communication_instructions').hide();
      });
  }, timeout_warning*1000);
}

function stopTimeout() {
  clearTimeout(timeout_timer);
}

function restartTimeout() {
  stopTimeout();
  startTimeout();
}

// Draw to and clear a canvas ---------------------------------------

function drawShepardCircle(angle, radius, canvas, mini_size) {
  var center_x = 240, center_y = 240;
  if (mini_size) {
    var center_x = 27.5, center_y = 27.5;
    radius *= 0.1145833333;
  }
  var x = radius * Math.cos(angle) + center_x;
  var y = radius * Math.sin(angle) + center_y;
  canvas.beginPath();
  canvas.arc(center_x, center_y, radius, 0, 2 * Math.PI, false);
  canvas.strokeStyle = 'black';
  canvas.stroke();
  canvas.beginPath();
  canvas.moveTo(center_x, center_y);
  canvas.lineTo(x, y);
  canvas.stroke();
}

function clearCanvas(canvas) {
  canvas.clearRect(0, 0, 480, 480);
}

// Display and reset canvases ---------------------------------------

function displayStimulusCanvas(stim_n) {
  var circle = shepard_circles[stim_n];
  drawShepardCircle(circle[0], circle[1], canvas_middle, false);
  $('#canvas_middle').show();
}

function resetStimulusCanvas() {
  $('#canvas_middle').hide();
  clearCanvas(canvas_middle);
}

function displaySignalCanvas(stim_n) {
  var circle = shepard_circles[stim_n];
  drawShepardCircle(circle[0], circle[1], canvas_left, false);
  $('#canvas_left').show();
}

function resetSignalCanvas() {
  $('#canvas_left').hide();
  clearCanvas(canvas_left);
}

function displayFeedbackCanvas(stim_n) {
  var circle = shepard_circles[stim_n];
  drawShepardCircle(circle[0], circle[1], canvas_right, false);
  $('#canvas_right').show();
}

function resetFeedbackCanvas() {
  $('#canvas_right').hide();
  clearCanvas(canvas_right);
}

function displayStimPicker() {
  $('#stim_picker').show();
  $('#canvas_right').show();
}

function resetStimPicker() {
  $('#stim_picker').hide();
  $('#canvas_right').hide();
  clearCanvas(canvas_right);
}

// Display an instruction -------------------------------------------

function displayInstruction(message, bonus) {
  if (bonus)
    message += '&nbsp;&nbsp;<span class="mini_bonus">+' + bonus + '&cent; if correct</span>';
  $('#message').html(message).show();
}

function resetPanel() {
  $('#message').hide().html('');
  $('#word_array').hide();
  $('#signal').hide();
  $('#spinner').show();
}

// Turn buttons and canvases green or red ---------------------------

function turnWordButtonGreen(button_id) {
  $('#word_button_' + button_id).css('background-color', '#67C200');
  setTimeout(function() {
    $('#word_button_' + button_id).css('background-color', '#03A7FF');
  }, 1000);
}

function turnWordButtonRed(button_id) {
  $('#word_button_' + button_id).css('background-color', '#FF2F00');
  setTimeout(function() {
    $('#word_button_' + button_id).css('background-color', '#03A7FF');
  }, 1000);
}

function turnCanvasGreen(canvas_id) {
  $('#stim_button_' + canvas_id).css('background-color', '#67C200');
  $('#stim_button_' + canvas_id).css('border', 'solid 1px #67C200');
  setTimeout(function() {
    $('#stim_button_' + canvas_id).css('background-color', '#FFFFFF');
    $('#stim_button_' + canvas_id).css('border', 'solid 1px #03A7FF');
  }, 1000);
}

function turnCanvasRed(canvas_id) {
  $('#stim_button_' + canvas_id).css('background-color', '#FF2F00');
  $('#stim_button_' + canvas_id).css('border', 'solid 1px #FF2F00');
  setTimeout(function() {
    $('#stim_button_' + canvas_id).css('background-color', '#FFFFFF');
    $('#stim_button_' + canvas_id).css('border', 'solid 1px #03A7FF');
  }, 1000);
}

// Randomize buttons or canvases ------------------------------------

function randomizeWordButtons() {
  var word_array = [0,1,2,3];
  shuffle(word_array);
  for (var i=0; i<word_array.length; i++) {
    $('#word_button_'+i).html(words[word_array[i]]);
  }
  return word_array;
}

function randomizeStimPicker() {
  var stim_array = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63];
  shuffle(stim_array);
  for (var i=0; i<stim_array.length; i++) {
    var circle = shepard_circles[stim_array[i]];
    comprehension_canvases[i].clearRect(0,0,55,55);
    drawShepardCircle(circle[0], circle[1], comprehension_canvases[i], true);
  }
  return stim_array;
}

// Cookies ----------------------------------------------------------

function writeTaskMarker(task_id) {
  var date = new Date();
  date.setTime(date.getTime() + 31536000000);
  document.cookie = 'task_id=' + task_id + '; expires=' + date.toGMTString() + '; path=/';
}

function readTaskMarker() {
  var cookies = document.cookie.split(';');
  for (var i=0; i<cookies.length; i++) {
    var cookie = cookies[i].split('=');
    if (cookie[0] === 'task_id')
      return cookie[1];
  }
  return null;
}

// Progress and money -----------------------------------------------

function updateProgress(progress) {
  var progress_width = $('#progress_bar').width();
  $('#progress').animate({width: progress * progress_width}, 250);
}

function updateMoney(money) {
  $('#money').html(money + '&cent;');
}

function updateWaitTime(wait) {
  var wait = Math.floor(wait);
  if (wait <= 0)
    $('#wait_time').html('...');
  else {
    var minutes = Math.floor(wait/60);
    var seconds = (wait % 60).toString();
    if (seconds.length === 1)
      seconds = '0' + seconds;
    $('#wait_time').html(minutes + ':' + seconds);
  }
}

// Display alerts ---------------------------------------------------

function displayAlert(message, callbackContinue, callbackExit) {
  $('#screen_blank').show();
  $('#alert_message').html(message);
  $('#alert_continue_button').click(function() {
    $('#alert_continue_button').off();
    $('#alert').hide();
    $('#screen_blank').hide();
    callbackContinue();
  });
  $('#alert_exit_button').click(function() {
    $('#alert_exit_button').off();
    $('#alert').hide();
    $('#screen_blank').hide();
    callbackExit();
  });
  $('#alert').show();
}

// ------------------------------------------------------------------
// Local event handlers
// ------------------------------------------------------------------

// Registration and consent button clicks ---------------------------

$('#submit_id').click(function() {
  user_id = $('#user_id').val().replace(/\D/g,'');
  $('#user_id').val(user_id);
  if (!id_validator.test(user_id))
    return $('#validation_message').html('Invalid Worker ID');
  $('#submit_id').hide();
  server.emit('register', { user_id });
});

$('#submit_consent').click(function() {
  if ($("#confirm_age").is(":checked") && $("#confirm_consent").is(":checked")) {
    $('#consent').hide();
    $('#training_instructions').show();
    restartTimeout();
  }
});

// Start button clicks ----------------------------------------------

$('#start_training').click(function() {
  $('#training_instructions').hide();
  $('#experiment').show();
  server.emit('start_training', { user_id });
  restartTimeout();
});

$('#start_production').click(function() {
  $('#production_instructions').hide();
  $('#experiment').show();
  $('#progress').css('width', 0);
  $('#progress_bar').css('width', 942);
  server.emit('production_start', { user_id });
  restartTimeout();
});

$('#start_comprehension').click(function() {
  $('#comprehension_instructions').hide();
  $('#experiment').show();
  $('#progress').css('width', 0);
  $('#progress_bar').css('width', 942);
  server.emit('comprehension_start', { user_id });
  restartTimeout();
});

$('#start_communication').click(function() {
  clearTimeout(timeout_timer);
  server.emit('communication_start', { user_id });
  $('#start_communication').hide();
  $('#communication_spinner').show();
  wait_estimation_timer = setInterval(function() {
    server.emit('communication_estimate_wait', { user_id });
  }, 13000);
});

// Comment button clicks --------------------------------------------

$('#send_comments').click(function() {
  var comments = $('#comments').val();
  if (comments.length > 0) {
    $('#comment_area').html('Thank you. Your comments have been submitted.');
    server.emit('send_comments', { user_id, comments });
  }
});

$('#early_send_comments').click(function() {
  var comments = $('#early_comments').val();
  if (comments.length > 0) {
    $('#early_comment_area').html('Thank you. Your comments have been submitted.');
    server.emit('send_comments', { user_id, comments });
  }
});

$('#partner_disconnect_send_comments').click(function() {
  var comments = $('#partner_disconnect_comments').val();
  if (comments.length > 0) {
    $('#partner_disconnect_comment_area').html('Thank you. Your comments have been submitted.');
    server.emit('send_comments', { user_id, comments });
  }
});

// Exit button clicks -----------------------------------------------

$('#terminate').click(function() {
  stopTimeout();
  displayAlert('Are you sure you want to exit the task? You will only receive 1¢ if you exit the task now.',
    function() {
      startTimeout();
    },
    function() {
      server.emit('terminate', { user_id });
      $('#experiment').hide();
    });
});

// ------------------------------------------------------------------
// Server event handlers
// ------------------------------------------------------------------

// Availability, registration, and consent form ---------------------

server.on('availability', function(payload) {
  if (payload.available) {
    $('#entry').show();
    $('#user_id').focus();
  } else
    $('#busy').show();
});

server.on('reject_id', function(payload) {
  $('#validation_message').html(payload.message);
  if (payload.cookie)
    writeTaskMarker(payload.cookie);
});

server.on('consent', function(payload) {
  words = payload.words;
  partition = payload.partition;
  switch (payload.test_type) {
    case 'production':
      $('#prod_training_image').show();
      break;
    case 'comprehension':
      $('#comp_training_image').show();
      break;
    case 'communication':
      $('#comm_training_image').show();
      break;
  }
  $('#entry').hide();
  $('#consent').show();
  writeTaskMarker(payload.cookie);
  startTimeout();
});

// Training trials --------------------------------------------------

server.on('training_trial', function(payload) {
  var word_array = randomizeWordButtons();
  var timer = Date.now();
  $('button[id^="word_button_"]').click(function() {
    $('button[id^="word_button_"]').off();
    restartTimeout();
    var reaction_time = Date.now() - timer;
    var button_id = parseInt($(this).attr('id').match(/word_button_(.)/)[1]);
    var selection = word_array[button_id];
    if (selection === payload.cat_n) {
      turnWordButtonGreen(button_id);
      displayInstruction('Correct!', false);
    } else {
      turnWordButtonGreen(word_array.indexOf(payload.cat_n));
      turnWordButtonRed(button_id);
      displayInstruction('Incorrect', false);
    }
    updateProgress(payload.progress);
    setTimeout(function() {
      resetStimulusCanvas();
      resetPanel();
      server.emit('training_response', { user_id, selection, button_id, reaction_time });
    }, 1000);
  });
  $('#spinner').hide();
  displayStimulusCanvas(payload.stim_n);
  setTimeout(function() {
    displayInstruction('This is a <strong>' + words[payload.cat_n] + '</strong>', false);
    setTimeout(function() {
      displayInstruction('What is it called?', false);
      $('#word_array').show();
    }, 2000);
  }, 1000);
});

// Mini-tests -------------------------------------------------------

server.on('production_mini_test', function(payload) {
  var word_array = randomizeWordButtons();
  var timer = Date.now();
  $('button[id^="word_button_"]').click(function() {
    $('button[id^="word_button_"]').off();
    restartTimeout();
    var reaction_time = Date.now() - timer;
    var button_id = parseInt($(this).attr('id').match(/word_button_(.)/)[1]);
    var selection = word_array[button_id];
    if (selection === payload.cat_n) {
      turnWordButtonGreen(button_id);
      displayInstruction('Correct!', false);
      updateMoney(payload.money + payload.bonus);
    } else {
      turnWordButtonGreen(word_array.indexOf(payload.cat_n));
      turnWordButtonRed(button_id);
      displayInstruction('Incorrect', false);
    }
    updateProgress(payload.progress);
    setTimeout(function() {
      resetStimulusCanvas();
      resetPanel();
      server.emit('training_response', { user_id, selection, button_id, reaction_time });
    }, 1000);
  });
  $('#spinner').hide();
  displayStimulusCanvas(payload.stim_n);
  displayInstruction('What is this called?', payload.bonus);
  setTimeout(function() {
    $('#word_array').show();
  }, 1000);
});

server.on('comprehension_mini_test', function(payload) {
  var stim_array = randomizeStimPicker();
  var timer = Date.now();
  $('canvas[id^="stim_button_"]').click(function() {
    $('canvas[id^="stim_button_"]').off();
    restartTimeout();
    var reaction_time = Date.now() - timer;
    var button_id = parseInt($(this).attr('id').match(/stim_button_(\d*)/)[1]);
    var selection = stim_array[button_id];
    if (payload.cat_n === partition[selection]) {
      turnCanvasGreen(button_id);
      displayInstruction('Correct!', false);
      updateMoney(payload.money + payload.bonus);
    } else {
      turnCanvasRed(button_id);
      displayInstruction('Incorrect', false);
    }
    updateProgress(payload.progress);
    setTimeout(function() {
      resetStimPicker();
      resetPanel();
      server.emit('training_response', { user_id, selection, button_id, reaction_time });
    }, 1000);
  })
  .mouseover(function() {
    $(this).css('background-color', '#03A7FF');
    var button_id = parseInt($(this).attr('id').match(/stim_button_(\d*)/)[1]);
    var circle = shepard_circles[stim_array[button_id]];
    drawShepardCircle(circle[0], circle[1], canvas_right, false);
  })
  .mouseout(function() {
    $(this).css('background-color', '#FFFFFF');
    clearCanvas(canvas_right);
  });
  $('#spinner').hide();
  displayInstruction('Click on a <strong>' + words[payload.cat_n] + '</strong>', payload.bonus);
  setTimeout(function() {
    displayStimPicker();
  }, 1000);
});

// Test instructions ------------------------------------------------

server.on('production_instructions', function(payload) {
  $('#money').hide();
  $('#experiment').hide();
  $('#production_instructions').show();
  $('#money').html('');
});

server.on('comprehension_instructions', function(payload) {
  $('#money').hide();
  $('#experiment').hide();
  $('#comprehension_instructions').show();
  $('#money').html('');
});

server.on('communication_instructions', function(payload) {
  $('#experiment').hide();
  $('#communication_instructions').show();
});

// Communication pairing --------------------------------------------

server.on('communication_wait', function(payload) {
  if (!wait_timer) {
    wait_timer = setInterval(function() {
      if (payload.wait_time > 0) {
        payload.wait_time -= 1;
        updateWaitTime(payload.wait_time);
      }
    }, 1000);
  }
}); 

server.on('communication_line_test', function() {
  console.log('Line test successful');
});

server.on('communication_paired', function(payload) {
  clearTimeout(wait_estimation_timer);
  clearTimeout(wait_timer);
  $('#communication_instructions').hide();
  $('#experiment').show();
  $('#progress').css('width', 0);
  startTimeout();
  server.emit('communication_next', { user_id });
});

// Production test --------------------------------------------------

server.on('production_trial', function(payload) {
  var word_array = randomizeWordButtons();
  timer = Date.now();
  $('button[id^="word_button_"]').click(function() {
    $('button[id^="word_button_"]').off();
    restartTimeout();
    var reaction_time = Date.now() - timer;
    var button_id = parseInt($(this).attr('id').match(/word_button_(.)/)[1]);
    var selection = word_array[button_id];
    resetStimulusCanvas();
    resetPanel();
    updateProgress(payload.progress);
    server.emit('production_response', { user_id, selection, button_id, reaction_time });
  });
  $('#spinner').hide();
  displayStimulusCanvas(payload.stim_n);
  displayInstruction('What is this called?', payload.bonus);
  setTimeout(function() {
    $('#word_array').show();
  }, 1000);
});

// Comprehension test -----------------------------------------------

server.on('comprehension_trial', function(payload) {
  var stim_array = randomizeStimPicker();
  var timer = Date.now();
  $('canvas[id^="stim_button_"]').click(function() {
    $('canvas[id^="stim_button_"]').off();
    restartTimeout();
    var reaction_time = Date.now() - timer;
    var button_id = parseInt($(this).attr('id').match(/stim_button_(\d*)/)[1]);
    var selection = stim_array[button_id];
    resetStimPicker();
    resetPanel();
    updateProgress(payload.progress);
    server.emit('comprehension_response', { user_id, selection, button_id, reaction_time });
  })
  .mouseover(function() {
    $(this).css('background-color', '#03A7FF');
    var button_id = parseInt($(this).attr('id').match(/stim_button_(\d*)/)[1]);
    var circle = shepard_circles[stim_array[button_id]];
    drawShepardCircle(circle[0], circle[1], canvas_right, false);
  })
  .mouseout(function() {
    $(this).css('background-color', '#FFFFFF');
    clearCanvas(canvas_right);
  });
  $('#spinner').hide();
  displayInstruction('Click on a <strong>' + words[payload.cat_n] + '</strong>', payload.bonus);
  setTimeout(function() {
    displayStimPicker();
  }, 1000);
});

// Communication test -----------------------------------------------

server.on('communication_trial', function(trial_payload) {
  if (trial_payload.director) {
    var word_array = randomizeWordButtons();
    var timer = Date.now();
    $('button[id^="word_button_"]').click(function() {
      $('button[id^="word_button_"]').off();
      restartTimeout();
      var reaction_time = Date.now() - timer;
      var button_id = parseInt($(this).attr('id').match(/word_button_(.)/)[1]);
      var selection = word_array[button_id];
      $('#word_array').hide();
      $('#spinner').show();
      resetStimulusCanvas();
      displaySignalCanvas(trial_payload.stim_n);
      displayInstruction('You called it <strong>' + words[selection] + '</strong> (waiting for your partner\'s response...)', false);
      server.emit('communication_signal', { user_id, selection, button_id, reaction_time });
    });
    server.on('communication_pass_feedback', function(payload) {
      server.off('communication_pass_feedback');
      updateProgress(trial_payload.progress);
      displayFeedbackCanvas(payload.feedback);
      if (payload.feedback === trial_payload.stim_n) {
        displayInstruction('Correct!', false);
        // updateMoney(payload.money + trial_payload.bonus);
      } else
        displayInstruction('Incorrect', false);
      setTimeout(function() {
        resetSignalCanvas();
        resetFeedbackCanvas();
        server.emit('communication_next', { user_id });
      }, 3000);
    });
    $('#spinner').hide();
    displayStimulusCanvas(trial_payload.stim_n);
    displayInstruction('What is this called?', trial_payload.bonus);
    setTimeout(function() {
      $('#word_array').show();
    }, 1000);
  } else {
    server.on('communication_pass_signal', function(payload) {
      server.off('communication_pass_signal');
      var stim_array = randomizeStimPicker();
      var timer = Date.now();
      $('canvas[id^="stim_button_"]').click(function() {
        $('canvas[id^="stim_button_"]').off();
        restartTimeout();
        var reaction_time = Date.now() - timer;
        var button_id = parseInt($(this).attr('id').match(/stim_button_(\d*)/)[1]);
        var selection = stim_array[button_id];
        server.emit('communication_feedback', { user_id, selection, button_id, reaction_time });
        if (selection === trial_payload.stim_n) {
          turnCanvasGreen(button_id);
          displayInstruction('Correct!', false);
          // updateMoney(payload.money + trial_payload.bonus);
        } else {
          turnCanvasGreen(stim_array.indexOf(trial_payload.stim_n));
          turnCanvasRed(button_id);
          displayInstruction('Incorrect', false);
        }
        updateProgress(trial_payload.progress);
        setTimeout(function() {
          resetStimPicker();
          resetPanel();
          server.emit('communication_next', { user_id });
        }, 3000);
      })
      .mouseover(function() {
        $(this).css('background-color', '#03A7FF');
        var button_id = parseInt($(this).attr('id').match(/stim_button_(\d*)/)[1]);
        var circle = shepard_circles[stim_array[button_id]];
        drawShepardCircle(circle[0], circle[1], canvas_right, false);
      })
      .mouseout(function() {
        $(this).css('background-color', '#FFFFFF');
        clearCanvas(canvas_right);
      });
      $('#spinner').hide();
      displayInstruction('Which picture is your partner communicating to you?', trial_payload.bonus);
      $('#signal').html(words[payload.signal]).show();
      setTimeout(function() {
        displayStimPicker();
      }, 1000);
    });
  }
});

// Ending, early-termination, and timeouts --------------------------

server.on('end', function(payload) {
  stopTimeout();
  $('#screen_blank').hide();
  $('#experiment').hide();
  $('#completion_code').html('<strong>Completion Code: ' + payload.completion_code + '</strong>');
  $('#basic_pay').html('$' + (payload.baseline_pay/100).toFixed(2));
  $('#training_bonus').html('$' + (payload.training_bonus/100).toFixed(2));
  $('#test_bonus').html('$' + (payload.test_bonus/100).toFixed(2));
  $('#total_pay').html('$' + (payload.total_pay/100).toFixed(2));
  $('#training_bonus_proportion').html((payload.training_bonus/2) + ' / 32');
  $('#test_bonus_proportion').html((payload.test_bonus/2) + ' / 64');
  $('#end').show();
  writeTaskMarker(payload.cookie);
});

server.on('early_exit', function(payload) {
  stopTimeout();
  $('#early_completion_code').html('<strong>Completion Code: ' + payload.completion_code + '</strong>');
  $('#early_exit').show();
  writeTaskMarker(payload.cookie);
});

server.on('partner_disconnect', function(payload) {
  stopTimeout();
  $('#screen_blank').hide();
  $('#experiment').hide();
  $('#partner_disconnect_completion_code').html('<strong>Completion Code: ' + payload.completion_code + '</strong>');
  $('#partner_disconnect_basic_pay').html('$' + (payload.baseline_pay/100).toFixed(2));
  $('#partner_disconnect_training_bonus').html('$' + (payload.training_bonus/100).toFixed(2));
  $('#partner_disconnect_test_bonus').html('$' + (payload.test_bonus/100).toFixed(2));
  $('#partner_disconnect_total_pay').html('$' + (payload.total_pay/100).toFixed(2));
  $('#partner_disconnect_training_bonus_proportion').html((payload.training_bonus/2) + ' / 32');
  $('#partner_disconnect_test_bonus_proportion').html((payload.test_bonus/2) + ' / 64');
  $('#partner_disconnect').show();
  writeTaskMarker(payload.cookie);
});

server.on('timeout', function(payload) {
  stopTimeout();
  $('#screen_blank').hide();
  $('#experiment').hide();
  $('#consent').hide();
  $('#training_instructions').hide();
  $('#production_instructions').hide();
  $('#comprehension_instructions').hide();
  $('#communication_instructions').hide();
  $('#timeout').show();
  writeTaskMarker(payload.cookie);
});

// ------------------------------------------------------------------
// On page initialization...
// ------------------------------------------------------------------

$(document).ready(function() {

// Check task availability ------------------------------------------

  if (readTaskMarker() === 'shepard')
    $('#unavailable').show();
  else
    server.emit('availability');

// Generate random ID for testing purposes --------------------------

  // var random_id = '';
  // for (i=0; i<8; i++) {
  //   random_id += Math.floor(Math.random() * 10);
  // }
  // $('#user_id').val(random_id);

});
