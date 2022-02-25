// > MODEL.START (Test)

// >> CLASS.START (P)

// VerCors verification instructions for SLCO class P.
class P {
    // Class variables.
    private final int[] x;
    private volatile int i;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.x, 1);
    ensures Perm(this.i, 1);

    // Require that the given values are not null.
    requires x != null;

    // Ensure that the right values are assigned.
    ensures this.x == x;
    ensures this.i == i;
    @*/
    P(int[] x, int i) {
        // Instantiate the class variables.
        this.x = x;
        this.i = i;
    }
}

// >>> STATE_MACHINE.START (SM1)

// VerCors verification instructions for SLCO state machine SM1.
class P_SM1Thread {
    // The class the state machine is a part of.
    private final P c;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    P_SM1Thread(P c) {
        // Reference to the parent SLCO class.
        this.c = c;
    }

    // SLCO expression wrapper | i >= 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_0_n_0() {
        //@ assume Perm(c.i, 1);
        return c.i >= 0;
    }

    // SLCO expression wrapper | i < 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_0_n_1() {
        //@ assume Perm(c.i, 1);
        return c.i < 2;
    }

    // SLCO expression wrapper | i >= 0 and i < 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_0_n_2() {
        return t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1();
    }

    // SLCO expression wrapper | x[i] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_0_n_3() {
        //@ assume Perm(c.i, 1);
        //@ assume 0 <= c.i && c.i < c.x.length;
        //@ assume Perm(c.x[c.i], 1);
        return c.x[c.i] == 0;
    }

    // SLCO expression wrapper | i >= 0 and i < 2 and x[i] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_0_n_4() {
        return t_SMC0_0_s_0_n_2() && t_SMC0_0_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_0() {
        // SLCO expression | i >= 0 and i < 2 and x[i] = 0.
        if(!(t_SMC0_0_s_0_n_4())) {
            return false;
        }
    }
}

// <<< STATE_MACHINE.END (SM1)

// >>> STATE_MACHINE.START (SM2)

// VerCors verification instructions for SLCO state machine SM2.
class P_SM2Thread {
    // The class the state machine is a part of.
    private final P c;

    // Thread local variables.
    private int[] y;
    private int j;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    P_SM2Thread(P c) {
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        y = new int[] { 0, 0 };
        j = 1;
    }

    // SLCO expression wrapper | i >= 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(y[*], 1);
    context Perm(j, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_0_n_0() {
        //@ assume Perm(c.i, 1);
        return c.i >= 0;
    }

    // SLCO expression wrapper | i < 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(y[*], 1);
    context Perm(j, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_0_n_1() {
        //@ assume Perm(c.i, 1);
        return c.i < 2;
    }

    // SLCO expression wrapper | i >= 0 and i < 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(y[*], 1);
    context Perm(j, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_0_n_2() {
        return t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1();
    }

    // SLCO expression wrapper | x[i] != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(y[*], 1);
    context Perm(j, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_0_n_3() {
        //@ assume Perm(c.i, 1);
        //@ assume 0 <= c.i && c.i < c.x.length;
        //@ assume Perm(c.x[c.i], 1);
        return c.x[c.i] != 0;
    }

    // SLCO expression wrapper | i >= 0 and i < 2 and x[i] != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(y[*], 1);
    context Perm(j, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_0_n_4() {
        return t_SMC0_0_s_0_n_2() && t_SMC0_0_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(y[*], 1);
    context Perm(j, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_0() {
        // SLCO expression | i >= 0 and i < 2 and x[i] != 0.
        if(!(t_SMC0_0_s_0_n_4())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(y[*], 1);
    context Perm(j, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_1() {
        // SLCO assignment | x[i] := y[i].
        //@ assume Perm(c.i, 1);
        //@ assume 0 <= c.i && c.i < c.x.length;
        //@ assume Perm(c.x[c.i], 1);
        //@ assume 0 <= c.i && c.i < y.length;
        c.x[c.i] = y[c.i];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(y[*], 1);
    context Perm(j, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_2() {
        // SLCO assignment | y[i] := 0.
        //@ assume Perm(c.i, 1);
        //@ assume 0 <= c.i && c.i < y.length;
        y[c.i] = 0;
    }
}

// <<< STATE_MACHINE.END (SM2)

// << CLASS.END (P)

// < MODEL.END (Test)