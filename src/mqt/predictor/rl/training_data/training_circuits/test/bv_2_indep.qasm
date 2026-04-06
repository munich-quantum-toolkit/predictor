OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[1];
x q[0];
measure q[1] -> c[0];
