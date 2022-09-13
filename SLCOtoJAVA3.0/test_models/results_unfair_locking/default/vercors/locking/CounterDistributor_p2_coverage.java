// > MODEL.START (CounterDistributor)

// >> CLASS.START (CounterDistributorExact)

// VerCors verification instructions for SLCO class CounterDistributorExact.
class CounterDistributorExact {
    // Class variables.
    private volatile int x;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.x, 1);

    // Ensure that the right values are assigned.
    ensures this.x == x;
    @*/
    CounterDistributorExact(int x) {
        // Instantiate the class variables.
        this.x = x;
    }
}

// >>> STATE_MACHINE.START (Counter)

// VerCors verification instructions for SLCO state machine Counter.
class CounterDistributorExact_CounterThread {
    // The class the state machine is a part of.
    private final CounterDistributorExact c;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    CounterDistributorExact_CounterThread(CounterDistributorExact c) {
        // Reference to the parent SLCO class.
        this.c = c;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_C_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_C_0_s_1() {
        // SLCO assignment | x := (x + 1) % 10.
        //@ assume Perm(c.x, 1);
        c.x = Math.floorMod((c.x + 1), 10);
    }
}

// <<< STATE_MACHINE.END (Counter)

// >>> STATE_MACHINE.START (Distributor)

// VerCors verification instructions for SLCO state machine Distributor.
class CounterDistributorExact_DistributorThread {
    // The class the state machine is a part of.
    private final CounterDistributorExact c;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    CounterDistributorExact_DistributorThread(CounterDistributorExact c) {
        // Reference to the parent SLCO class.
        this.c = c;
    }

    // SLCO expression wrapper | x = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_0_s_0_n_0() {
        //@ assume Perm(c.x, 1);
        return c.x == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_0_s_0() {
        // SLCO expression | x = 0.
        if(!(t_P_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | x = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_1_s_0_n_0() {
        //@ assume Perm(c.x, 1);
        return c.x == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_1_s_0() {
        // SLCO expression | x = 1.
        if(!(t_P_1_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | x = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_2_s_0_n_0() {
        //@ assume Perm(c.x, 1);
        return c.x == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_2_s_0() {
        // SLCO expression | x = 2.
        if(!(t_P_2_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | x = 3.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_3_s_0_n_0() {
        //@ assume Perm(c.x, 1);
        return c.x == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_3_s_0() {
        // SLCO expression | x = 3.
        if(!(t_P_3_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | x = 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_4_s_0_n_0() {
        //@ assume Perm(c.x, 1);
        return c.x == 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_4_s_0() {
        // SLCO expression | x = 4.
        if(!(t_P_4_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | x = 5.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_5_s_0_n_0() {
        //@ assume Perm(c.x, 1);
        return c.x == 5;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_5_s_0() {
        // SLCO expression | x = 5.
        if(!(t_P_5_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | x = 6.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_6_s_0_n_0() {
        //@ assume Perm(c.x, 1);
        return c.x == 6;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_6_s_0() {
        // SLCO expression | x = 6.
        if(!(t_P_6_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | x = 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_7_s_0_n_0() {
        //@ assume Perm(c.x, 1);
        return c.x == 7;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_7_s_0() {
        // SLCO expression | x = 7.
        if(!(t_P_7_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | x = 8.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_8_s_0_n_0() {
        //@ assume Perm(c.x, 1);
        return c.x == 8;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_8_s_0() {
        // SLCO expression | x = 8.
        if(!(t_P_8_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | x = 9.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_9_s_0_n_0() {
        //@ assume Perm(c.x, 1);
        return c.x == 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);
    @*/
    private boolean t_P_9_s_0() {
        // SLCO expression | x = 9.
        if(!(t_P_9_s_0_n_0())) {
            return false;
        }
    }
}

// <<< STATE_MACHINE.END (Distributor)

// << CLASS.END (CounterDistributorExact)

// < MODEL.END (CounterDistributor)