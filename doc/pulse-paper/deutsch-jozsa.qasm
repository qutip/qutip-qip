OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
x q[2];
h q; 
cx q[0], q[2];
cx q[1], q[2];
h q[0];
h q[1];