
// Name of Experiment: W-state v1

OPENQASM 2.0;
include "qelib1.inc";


qreg q[4]; // This comments also should be handled.
creg c[3]; // QASMBench has this type of comments.
gate cH a,b {
h b;
sdg b;
cx a,b;
h b;
t b;
cx a,b;
t b;
h b;
s b;
x b;
s a;
}

u3(1.91063,0,0) q[0];
cH q[0],q[1];
ccx q[0],q[1],q[2];
x q[0];
x q[1];
cx q[0],q[1];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
