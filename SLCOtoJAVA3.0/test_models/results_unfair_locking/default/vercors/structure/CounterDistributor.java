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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private void range_check_assumption_t_0_s_2() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(true);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (c.x == _rhs_0);
    ensures !_guard ==> (c.x == \old(c.x));
    @*/
    // SLCO transition (p:0, id:0) | C -> C | true | x := (x + 1) % 10.
    private boolean execute_transition_C_0() {
        // (Superfluous) SLCO expression | true.
        //@ ghost _guard = true;

        // SLCO assignment | x := (x + 1) % 10.
        range_check_assumption_t_0_s_2();
        //@ ghost _rhs_0 = Math.floorMod((c.x + 1), 10);
        c.x = Math.floorMod((c.x + 1), 10);
        //@ assert c.x == _rhs_0;

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);
    @*/
    // Attempt to fire a transition starting in state C.
    private void exec_C() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | C -> C | true | x := (x + 1) % 10.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_C_0()) {
            return;
        }
        // [SEQ.END]
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x == 0);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private boolean t_P_0_s_0_n_0() {
        return c.x == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.x == 0);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.x == \old(c.x);
    @*/
    // SLCO transition (p:0, id:0) | P -> P | x = 0.
    private boolean execute_transition_P_0() {
        // SLCO expression | x = 0.
        //@ ghost _guard = c.x == 0;
        if(!(t_P_0_s_0_n_0())) {
            //@ assert !(c.x == 0);
            return false;
        }
        //@ assert c.x == 0;

        return true;
    }

    // SLCO expression wrapper | x = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private boolean t_P_1_s_0_n_0() {
        return c.x == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private void range_check_assumption_t_1() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.x == 1);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.x == \old(c.x);
    @*/
    // SLCO transition (p:0, id:1) | P -> P | x = 1.
    private boolean execute_transition_P_1() {
        // SLCO expression | x = 1.
        //@ ghost _guard = c.x == 1;
        if(!(t_P_1_s_0_n_0())) {
            //@ assert !(c.x == 1);
            return false;
        }
        //@ assert c.x == 1;

        return true;
    }

    // SLCO expression wrapper | x = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private boolean t_P_2_s_0_n_0() {
        return c.x == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private void range_check_assumption_t_2() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.x == 2);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.x == \old(c.x);
    @*/
    // SLCO transition (p:0, id:2) | P -> P | x = 2.
    private boolean execute_transition_P_2() {
        // SLCO expression | x = 2.
        //@ ghost _guard = c.x == 2;
        if(!(t_P_2_s_0_n_0())) {
            //@ assert !(c.x == 2);
            return false;
        }
        //@ assert c.x == 2;

        return true;
    }

    // SLCO expression wrapper | x = 3.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x == 3);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private boolean t_P_3_s_0_n_0() {
        return c.x == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private void range_check_assumption_t_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.x == 3);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.x == \old(c.x);
    @*/
    // SLCO transition (p:0, id:3) | P -> P | x = 3.
    private boolean execute_transition_P_3() {
        // SLCO expression | x = 3.
        //@ ghost _guard = c.x == 3;
        if(!(t_P_3_s_0_n_0())) {
            //@ assert !(c.x == 3);
            return false;
        }
        //@ assert c.x == 3;

        return true;
    }

    // SLCO expression wrapper | x = 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x == 4);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private boolean t_P_4_s_0_n_0() {
        return c.x == 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private void range_check_assumption_t_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.x == 4);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.x == \old(c.x);
    @*/
    // SLCO transition (p:0, id:4) | P -> P | x = 4.
    private boolean execute_transition_P_4() {
        // SLCO expression | x = 4.
        //@ ghost _guard = c.x == 4;
        if(!(t_P_4_s_0_n_0())) {
            //@ assert !(c.x == 4);
            return false;
        }
        //@ assert c.x == 4;

        return true;
    }

    // SLCO expression wrapper | x = 5.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x == 5);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private boolean t_P_5_s_0_n_0() {
        return c.x == 5;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private void range_check_assumption_t_5() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.x == 5);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.x == \old(c.x);
    @*/
    // SLCO transition (p:0, id:5) | P -> P | x = 5.
    private boolean execute_transition_P_5() {
        // SLCO expression | x = 5.
        //@ ghost _guard = c.x == 5;
        if(!(t_P_5_s_0_n_0())) {
            //@ assert !(c.x == 5);
            return false;
        }
        //@ assert c.x == 5;

        return true;
    }

    // SLCO expression wrapper | x = 6.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x == 6);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private boolean t_P_6_s_0_n_0() {
        return c.x == 6;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private void range_check_assumption_t_6() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.x == 6);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.x == \old(c.x);
    @*/
    // SLCO transition (p:0, id:6) | P -> P | x = 6.
    private boolean execute_transition_P_6() {
        // SLCO expression | x = 6.
        //@ ghost _guard = c.x == 6;
        if(!(t_P_6_s_0_n_0())) {
            //@ assert !(c.x == 6);
            return false;
        }
        //@ assert c.x == 6;

        return true;
    }

    // SLCO expression wrapper | x = 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x == 7);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private boolean t_P_7_s_0_n_0() {
        return c.x == 7;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private void range_check_assumption_t_7() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.x == 7);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.x == \old(c.x);
    @*/
    // SLCO transition (p:0, id:7) | P -> P | x = 7.
    private boolean execute_transition_P_7() {
        // SLCO expression | x = 7.
        //@ ghost _guard = c.x == 7;
        if(!(t_P_7_s_0_n_0())) {
            //@ assert !(c.x == 7);
            return false;
        }
        //@ assert c.x == 7;

        return true;
    }

    // SLCO expression wrapper | x = 8.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x == 8);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private boolean t_P_8_s_0_n_0() {
        return c.x == 8;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private void range_check_assumption_t_8() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.x == 8);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.x == \old(c.x);
    @*/
    // SLCO transition (p:0, id:8) | P -> P | x = 8.
    private boolean execute_transition_P_8() {
        // SLCO expression | x = 8.
        //@ ghost _guard = c.x == 8;
        if(!(t_P_8_s_0_n_0())) {
            //@ assert !(c.x == 8);
            return false;
        }
        //@ assert c.x == 8;

        return true;
    }

    // SLCO expression wrapper | x = 9.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x == 9);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private boolean t_P_9_s_0_n_0() {
        return c.x == 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.x == \old(c.x);
    @*/
    private void range_check_assumption_t_9() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.x == 9);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.x == \old(c.x);
    @*/
    // SLCO transition (p:0, id:9) | P -> P | x = 9.
    private boolean execute_transition_P_9() {
        // SLCO expression | x = 9.
        //@ ghost _guard = c.x == 9;
        if(!(t_P_9_s_0_n_0())) {
            //@ assert !(c.x == 9);
            return false;
        }
        //@ assert c.x == 9;

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x, 1);
    @*/
    // Attempt to fire a transition starting in state P.
    private void exec_P() {
        // [SEQ.START]
        // [DET.START]
        // SLCO transition (p:0, id:0) | P -> P | x = 0.
        //@ ghost range_check_assumption_t_9();
        if(execute_transition_P_0()) {
            return;
        }
        // SLCO transition (p:0, id:1) | P -> P | x = 1.
        //@ ghost range_check_assumption_t_9();
        if(execute_transition_P_1()) {
            return;
        }
        // SLCO transition (p:0, id:2) | P -> P | x = 2.
        //@ ghost range_check_assumption_t_9();
        if(execute_transition_P_2()) {
            return;
        }
        // SLCO transition (p:0, id:3) | P -> P | x = 3.
        //@ ghost range_check_assumption_t_9();
        if(execute_transition_P_3()) {
            return;
        }
        // SLCO transition (p:0, id:4) | P -> P | x = 4.
        //@ ghost range_check_assumption_t_9();
        if(execute_transition_P_4()) {
            return;
        }
        // SLCO transition (p:0, id:5) | P -> P | x = 5.
        //@ ghost range_check_assumption_t_9();
        if(execute_transition_P_5()) {
            return;
        }
        // SLCO transition (p:0, id:6) | P -> P | x = 6.
        //@ ghost range_check_assumption_t_9();
        if(execute_transition_P_6()) {
            return;
        }
        // SLCO transition (p:0, id:7) | P -> P | x = 7.
        //@ ghost range_check_assumption_t_9();
        if(execute_transition_P_7()) {
            return;
        }
        // SLCO transition (p:0, id:8) | P -> P | x = 8.
        //@ ghost range_check_assumption_t_9();
        if(execute_transition_P_8()) {
            return;
        }
        // SLCO transition (p:0, id:9) | P -> P | x = 9.
        //@ ghost range_check_assumption_t_9();
        if(execute_transition_P_9()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }
}

// <<< STATE_MACHINE.END (Distributor)

// << CLASS.END (CounterDistributorExact)

// < MODEL.END (CounterDistributor)