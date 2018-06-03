<?php

$username = $_GET['username'];
$iteration = $_GET['iteration'];

if (preg_match('/^\d+$/', $iteration)) {
	$out = shell_exec("ssh -i /var/www/.ssh/id_rsa s1153197@blake.ppls.ed.ac.uk \"./qnext.py $username $iteration\"");
}

?>