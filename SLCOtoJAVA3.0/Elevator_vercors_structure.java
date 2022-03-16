// > MODEL.START (Elevator)

// >> CLASS.START (GlobalClass)

// VerCors verification instructions for SLCO class GlobalClass.
class GlobalClass {
    // Class variables.
    private final char[] req;
    private volatile int t;
    private volatile int p;
    private volatile char v;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.req, 1);
    ensures Perm(this.t, 1);
    ensures Perm(this.p, 1);
    ensures Perm(this.v, 1);

    // Require that the given values are not null.
    requires req != null;

    // Ensure that the right values are assigned.
    ensures this.req == req;
    ensures this.t == t;
    ensures this.p == p;
    ensures this.v == v;
    @*/
    GlobalClass(char[] req, int t, int p, char v) {
        // Instantiate the class variables.
        this.req = req;
        this.t = t;
        this.p = p;
        this.v = v;
    }
}

// >>> STATE_MACHINE.START (cabin)

// VerCors verification instructions for SLCO state machine cabin.
class GlobalClass_cabinThread {
    // The class the state machine is a part of.
    private final GlobalClass c;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    GlobalClass_cabinThread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;
    }

    // SLCO expression wrapper | v > 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.v > 0);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_idle_0_s_0_n_0() {
        return c.v > 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.v > 0);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    // SLCO transition (p:0, id:0) | idle -> mov | v > 0.
    private boolean execute_transition_idle_0() {
        // SLCO expression | v > 0.
        //@ ghost _guard = c.v > 0;
        if(!(t_idle_0_s_0_n_0())) {
            //@ assert !(c.v > 0);
            return false;
        }
        //@ assert c.v > 0;

        return true;
    }

    // SLCO expression wrapper | t = p.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t == c.p);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_mov_0_s_0_n_0() {
        return c.t == c.p;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.t == c.p);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    // SLCO transition (p:0, id:0) | mov -> open | t = p.
    private boolean execute_transition_mov_0() {
        // SLCO expression | t = p.
        //@ ghost _guard = c.t == c.p;
        if(!(t_mov_0_s_0_n_0())) {
            //@ assert !(c.t == c.p);
            return false;
        }
        //@ assert c.t == c.p;

        return true;
    }

    // SLCO expression wrapper | t < p.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t < c.p);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_mov_1_s_0_n_0() {
        return c.t < c.p;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_1_s_2() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_1() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.t < c.p);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures _guard ==> (c.p == _rhs_0);
    ensures !_guard ==> (c.p == \old(c.p));
    ensures c.v == \old(c.v);
    @*/
    // SLCO transition (p:0, id:1) | mov -> mov | [t < p; p := p - 1].
    private boolean execute_transition_mov_1() {
        // SLCO composite | [t < p; p := p - 1].
        // SLCO expression | t < p.
        //@ ghost _guard = c.t < c.p;
        if(!(t_mov_1_s_0_n_0())) {
            //@ assert !(c.t < c.p);
            return false;
        }
        //@ assert c.t < c.p;
        // SLCO assignment | p := p - 1.
        range_check_assumption_t_1_s_2();
        //@ ghost _rhs_0 = c.p - 1;
        c.p = c.p - 1;
        //@ assert c.p == _rhs_0;

        return true;
    }

    // SLCO expression wrapper | t > p.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t > c.p);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_mov_2_s_0_n_0() {
        return c.t > c.p;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_2_s_2() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_2() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.t > c.p);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures _guard ==> (c.p == _rhs_0);
    ensures !_guard ==> (c.p == \old(c.p));
    ensures c.v == \old(c.v);
    @*/
    // SLCO transition (p:0, id:2) | mov -> mov | [t > p; p := p + 1].
    private boolean execute_transition_mov_2() {
        // SLCO composite | [t > p; p := p + 1].
        // SLCO expression | t > p.
        //@ ghost _guard = c.t > c.p;
        if(!(t_mov_2_s_0_n_0())) {
            //@ assert !(c.t > c.p);
            return false;
        }
        //@ assert c.t > c.p;
        // SLCO assignment | p := p + 1.
        range_check_assumption_t_2_s_2();
        //@ ghost _rhs_0 = c.p + 1;
        c.p = c.p + 1;
        //@ assert c.p == _rhs_0;

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.p && c.p < 4;

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_0_s_3() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.p && c.p < 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_0_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    pure int value_open_0_req(int _i, int _index_0, int _rhs_0, int v_old) = (_i == _index_0) ? _rhs_0 : v_old;
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(true);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;
    yields int _index_0;
    yields int _rhs_1;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == value_open_0_req(_i, _index_0, _rhs_0, \old(c.req[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures _guard ==> (c.v == _rhs_1);
    ensures !_guard ==> (c.v == \old(c.v));
    @*/
    // SLCO transition (p:0, id:0) | open -> idle | true | [true; req[p] := 0; v := 0].
    private boolean execute_transition_open_0() {
        // (Superfluous) SLCO expression | true.
        //@ ghost _guard = true;

        // SLCO composite | [req[p] := 0; v := 0] -> [true; req[p] := 0; v := 0].
        // (Superfluous) SLCO expression | true.
        //@ ghost _guard = true;
        // SLCO assignment | req[p] := 0.
        range_check_assumption_t_0_s_3();
        //@ ghost _rhs_0 = 0;
        //@ ghost _index_0 = c.p;
        c.req[c.p] = (0) & 0xff;
        //@ assert c.req[_index_0] == _rhs_0;
        // SLCO assignment | v := 0.
        range_check_assumption_t_0_s_4();
        //@ ghost _rhs_1 = 0;
        c.v = (0) & 0xff;
        //@ assert c.v == _rhs_1;

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);
    @*/
    // Attempt to fire a transition starting in state idle.
    private void exec_idle() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | idle -> mov | v > 0.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_idle_0()) {
            return;
        }
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);
    @*/
    // Attempt to fire a transition starting in state mov.
    private void exec_mov() {
        // [SEQ.START]
        // [DET.START]
        // SLCO transition (p:0, id:0) | mov -> open | t = p.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_mov_0()) {
            return;
        }
        // SLCO transition (p:0, id:1) | mov -> mov | [t < p; p := p - 1].
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_mov_1()) {
            return;
        }
        // SLCO transition (p:0, id:2) | mov -> mov | [t > p; p := p + 1].
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_mov_2()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);
    @*/
    // Attempt to fire a transition starting in state open.
    private void exec_open() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | open -> idle | true | [true; req[p] := 0; v := 0].
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_open_0()) {
            return;
        }
        // [SEQ.END]
    }
}

// <<< STATE_MACHINE.END (cabin)

// >>> STATE_MACHINE.START (environment)

// VerCors verification instructions for SLCO state machine environment.
class GlobalClass_environmentThread {
    // The class the state machine is a part of.
    private final GlobalClass c;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    GlobalClass_environmentThread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;
    }

    // SLCO expression wrapper | req[0] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 0 && 0 < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.req[0] == 0);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_read_0_s_0_n_0() {
        return c.req[0] == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 0 && 0 < 4;

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_0_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 0 && 0 < 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 0 && 0 < 4;

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 0 && 0 < 4;
    }

    /*@
    pure int value_read_0_req(int _i, int _index_0, int _rhs_0, int v_old) = (_i == _index_0) ? _rhs_0 : v_old;
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 0 && 0 < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.req[0] == 0);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;
    yields int _index_0;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == value_read_0_req(_i, _index_0, _rhs_0, \old(c.req[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    // SLCO transition (p:0, id:0) | read -> read | [req[0] = 0; req[0] := 1].
    private boolean execute_transition_read_0() {
        // SLCO composite | [req[0] = 0; req[0] := 1].
        // SLCO expression | req[0] = 0.
        //@ ghost _guard = c.req[0] == 0;
        if(!(t_read_0_s_0_n_0())) {
            //@ assert !(c.req[0] == 0);
            return false;
        }
        //@ assert c.req[0] == 0;
        // SLCO assignment | req[0] := 1.
        range_check_assumption_t_0_s_2();
        //@ ghost _rhs_0 = 1;
        //@ ghost _index_0 = 0;
        c.req[0] = (1) & 0xff;
        //@ assert c.req[_index_0] == _rhs_0;

        return true;
    }

    // SLCO expression wrapper | req[1] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 1 && 1 < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.req[1] == 0);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_read_1_s_0_n_0() {
        return c.req[1] == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 1 && 1 < 4;

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_1_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 1 && 1 < 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 1 && 1 < 4;

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 1 && 1 < 4;
    }

    /*@
    pure int value_read_1_req(int _i, int _index_0, int _rhs_0, int v_old) = (_i == _index_0) ? _rhs_0 : v_old;
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 1 && 1 < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.req[1] == 0);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;
    yields int _index_0;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == value_read_1_req(_i, _index_0, _rhs_0, \old(c.req[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    // SLCO transition (p:0, id:1) | read -> read | [req[1] = 0; req[1] := 1].
    private boolean execute_transition_read_1() {
        // SLCO composite | [req[1] = 0; req[1] := 1].
        // SLCO expression | req[1] = 0.
        //@ ghost _guard = c.req[1] == 0;
        if(!(t_read_1_s_0_n_0())) {
            //@ assert !(c.req[1] == 0);
            return false;
        }
        //@ assert c.req[1] == 0;
        // SLCO assignment | req[1] := 1.
        range_check_assumption_t_1_s_2();
        //@ ghost _rhs_0 = 1;
        //@ ghost _index_0 = 1;
        c.req[1] = (1) & 0xff;
        //@ assert c.req[_index_0] == _rhs_0;

        return true;
    }

    // SLCO expression wrapper | req[2] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 2 && 2 < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.req[2] == 0);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_read_2_s_0_n_0() {
        return c.req[2] == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 2 && 2 < 4;

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_2_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 2 && 2 < 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 2 && 2 < 4;

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 2 && 2 < 4;
    }

    /*@
    pure int value_read_2_req(int _i, int _index_0, int _rhs_0, int v_old) = (_i == _index_0) ? _rhs_0 : v_old;
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 2 && 2 < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.req[2] == 0);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;
    yields int _index_0;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == value_read_2_req(_i, _index_0, _rhs_0, \old(c.req[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    // SLCO transition (p:0, id:2) | read -> read | [req[2] = 0; req[2] := 1].
    private boolean execute_transition_read_2() {
        // SLCO composite | [req[2] = 0; req[2] := 1].
        // SLCO expression | req[2] = 0.
        //@ ghost _guard = c.req[2] == 0;
        if(!(t_read_2_s_0_n_0())) {
            //@ assert !(c.req[2] == 0);
            return false;
        }
        //@ assert c.req[2] == 0;
        // SLCO assignment | req[2] := 1.
        range_check_assumption_t_2_s_2();
        //@ ghost _rhs_0 = 1;
        //@ ghost _index_0 = 2;
        c.req[2] = (1) & 0xff;
        //@ assert c.req[_index_0] == _rhs_0;

        return true;
    }

    // SLCO expression wrapper | req[3] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 3 && 3 < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.req[3] == 0);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_read_3_s_0_n_0() {
        return c.req[3] == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 3 && 3 < 4;

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_3_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 3 && 3 < 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 3 && 3 < 4;

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_3() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 3 && 3 < 4;
    }

    /*@
    pure int value_read_3_req(int _i, int _index_0, int _rhs_0, int v_old) = (_i == _index_0) ? _rhs_0 : v_old;
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 3 && 3 < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.req[3] == 0);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;
    yields int _index_0;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == value_read_3_req(_i, _index_0, _rhs_0, \old(c.req[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    // SLCO transition (p:0, id:3) | read -> read | [req[3] = 0; req[3] := 1].
    private boolean execute_transition_read_3() {
        // SLCO composite | [req[3] = 0; req[3] := 1].
        // SLCO expression | req[3] = 0.
        //@ ghost _guard = c.req[3] == 0;
        if(!(t_read_3_s_0_n_0())) {
            //@ assert !(c.req[3] == 0);
            return false;
        }
        //@ assert c.req[3] == 0;
        // SLCO assignment | req[3] := 1.
        range_check_assumption_t_3_s_2();
        //@ ghost _rhs_0 = 1;
        //@ ghost _index_0 = 3;
        c.req[3] = (1) & 0xff;
        //@ assert c.req[_index_0] == _rhs_0;

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);
    @*/
    // Attempt to fire a transition starting in state read.
    private void exec_read() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | read -> read | [req[0] = 0; req[0] := 1].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_read_0()) {
            return;
        }
        // SLCO transition (p:0, id:1) | read -> read | [req[1] = 0; req[1] := 1].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_read_1()) {
            return;
        }
        // SLCO transition (p:0, id:2) | read -> read | [req[2] = 0; req[2] := 1].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_read_2()) {
            return;
        }
        // SLCO transition (p:0, id:3) | read -> read | [req[3] = 0; req[3] := 1].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_read_3()) {
            return;
        }
        // [SEQ.END]
    }
}

// <<< STATE_MACHINE.END (environment)

// >>> STATE_MACHINE.START (controller)

// VerCors verification instructions for SLCO state machine controller.
class GlobalClass_controllerThread {
    // The class the state machine is a part of.
    private final GlobalClass c;

    // Thread local variables.
    private char ldir;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    GlobalClass_controllerThread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        ldir = (char) 0;
    }

    // SLCO expression wrapper | v = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.v == 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_wait_0_s_0_n_0() {
        return c.v == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_0_s_2() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.v == 0);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures _guard ==> (c.t == _rhs_0);
    ensures !_guard ==> (c.t == \old(c.t));
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    ensures ldir == \old(ldir);
    @*/
    // SLCO transition (p:0, id:0) | wait -> work | [v = 0; t := t + (2 * ldir) - 1].
    private boolean execute_transition_wait_0() {
        // SLCO composite | [v = 0; t := t + (2 * ldir) - 1].
        // SLCO expression | v = 0.
        //@ ghost _guard = c.v == 0;
        if(!(t_wait_0_s_0_n_0())) {
            //@ assert !(c.v == 0);
            return false;
        }
        //@ assert c.v == 0;
        // SLCO assignment | t := t + (2 * ldir) - 1.
        range_check_assumption_t_0_s_2();
        //@ ghost _rhs_0 = c.t + (2 * ldir) - 1;
        c.t = c.t + (2 * ldir) - 1;
        //@ assert c.t == _rhs_0;

        return true;
    }

    // SLCO expression wrapper | t < 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t < 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_work_0_s_0_n_0() {
        return c.t < 0;
    }

    // SLCO expression wrapper | t = 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t == 4);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_work_0_s_0_n_1() {
        return c.t == 4;
    }

    // SLCO expression wrapper | t < 0 or t = 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t < 0 || c.t == 4);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_work_0_s_0_n_2() {
        return t_work_0_s_0_n_0() || t_work_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_0_s_2() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.t < 0 || c.t == 4);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    ensures _guard ==> (ldir == _rhs_0);
    ensures !_guard ==> (ldir == \old(ldir));
    @*/
    // SLCO transition (p:0, id:0) | work -> wait | [t < 0 or t = 4; ldir := 1 - ldir].
    private boolean execute_transition_work_0() {
        // SLCO composite | [t < 0 or t = 4; ldir := 1 - ldir].
        // SLCO expression | t < 0 or t = 4.
        //@ ghost _guard = c.t < 0 || c.t == 4;
        if(!(t_work_0_s_0_n_2())) {
            //@ assert !(c.t < 0 || c.t == 4);
            return false;
        }
        //@ assert c.t < 0 || c.t == 4;
        // SLCO assignment | ldir := 1 - ldir.
        range_check_assumption_t_0_s_2();
        //@ ghost _rhs_0 = 1 - ldir;
        ldir = (1 - ldir) & 0xff;
        //@ assert ldir == _rhs_0;

        return true;
    }

    // SLCO expression wrapper | t >= 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t >= 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_work_1_s_0_n_0() {
        return c.t >= 0;
    }

    // SLCO expression wrapper | t < 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t < 4);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_work_1_s_0_n_1() {
        return c.t < 4;
    }

    // SLCO expression wrapper | t >= 0 and t < 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t >= 0 && c.t < 4);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_work_1_s_0_n_2() {
        return t_work_1_s_0_n_0() && t_work_1_s_0_n_1();
    }

    // SLCO expression wrapper | req[t] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.t && c.t < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.req[c.t] == 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_work_1_s_0_n_3() {
        return c.req[c.t] == 1;
    }

    // SLCO expression wrapper | t >= 0 and t < 4 and req[t] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.t && c.t < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t >= 0 && c.t < 4 && c.req[c.t] == 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_work_1_s_0_n_4() {
        return t_work_1_s_0_n_2() && t_work_1_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.t && c.t < 4;

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.t && c.t < 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.t && c.t < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.t >= 0 && c.t < 4 && c.req[c.t] == 1);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    ensures ldir == \old(ldir);
    @*/
    // SLCO transition (p:0, id:1) | work -> done | t >= 0 and t < 4 and req[t] = 1.
    private boolean execute_transition_work_1() {
        // SLCO expression | t >= 0 and t < 4 and req[t] = 1.
        //@ ghost _guard = c.t >= 0 && c.t < 4 && c.req[c.t] == 1;
        if(!(t_work_1_s_0_n_4())) {
            //@ assert !(c.t >= 0 && c.t < 4 && c.req[c.t] == 1);
            return false;
        }
        //@ assert c.t >= 0 && c.t < 4 && c.req[c.t] == 1;

        return true;
    }

    // SLCO expression wrapper | t >= 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t >= 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_work_2_s_0_n_0() {
        return c.t >= 0;
    }

    // SLCO expression wrapper | t < 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t < 4);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_work_2_s_0_n_1() {
        return c.t < 4;
    }

    // SLCO expression wrapper | t >= 0 and t < 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t >= 0 && c.t < 4);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_work_2_s_0_n_2() {
        return t_work_2_s_0_n_0() && t_work_2_s_0_n_1();
    }

    // SLCO expression wrapper | req[t] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.t && c.t < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.req[c.t] == 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_work_2_s_0_n_3() {
        return c.req[c.t] == 0;
    }

    // SLCO expression wrapper | t >= 0 and t < 4 and req[t] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.t && c.t < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.t >= 0 && c.t < 4 && c.req[c.t] == 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private boolean t_work_2_s_0_n_4() {
        return t_work_2_s_0_n_2() && t_work_2_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_2_s_2() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.t && c.t < 4;

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.t && c.t < 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.t && c.t < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.t >= 0 && c.t < 4 && c.req[c.t] == 0);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures _guard ==> (c.t == _rhs_0);
    ensures !_guard ==> (c.t == \old(c.t));
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    ensures ldir == \old(ldir);
    @*/
    // SLCO transition (p:0, id:2) | work -> work | [t >= 0 and t < 4 and req[t] = 0; t := t + (2 * ldir) - 1].
    private boolean execute_transition_work_2() {
        // SLCO composite | [t >= 0 and t < 4 and req[t] = 0; t := t + (2 * ldir) - 1].
        // SLCO expression | t >= 0 and t < 4 and req[t] = 0.
        //@ ghost _guard = c.t >= 0 && c.t < 4 && c.req[c.t] == 0;
        if(!(t_work_2_s_0_n_4())) {
            //@ assert !(c.t >= 0 && c.t < 4 && c.req[c.t] == 0);
            return false;
        }
        //@ assert c.t >= 0 && c.t < 4 && c.req[c.t] == 0;
        // SLCO assignment | t := t + (2 * ldir) - 1.
        range_check_assumption_t_2_s_2();
        //@ ghost _rhs_0 = c.t + (2 * ldir) - 1;
        c.t = c.t + (2 * ldir) - 1;
        //@ assert c.t == _rhs_0;

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_0_s_2() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(true);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures _guard ==> (c.v == _rhs_0);
    ensures !_guard ==> (c.v == \old(c.v));
    ensures ldir == \old(ldir);
    @*/
    // SLCO transition (p:0, id:0) | done -> wait | true | v := 1.
    private boolean execute_transition_done_0() {
        // (Superfluous) SLCO expression | true.
        //@ ghost _guard = true;

        // SLCO assignment | [v := 1] -> v := 1.
        range_check_assumption_t_0_s_2();
        //@ ghost _rhs_0 = 1;
        c.v = (1) & 0xff;
        //@ assert c.v == _rhs_0;

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);
    @*/
    // Attempt to fire a transition starting in state wait.
    private void exec_wait() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | wait -> work | [v = 0; t := t + (2 * ldir) - 1].
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_wait_0()) {
            return;
        }
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);
    @*/
    // Attempt to fire a transition starting in state work.
    private void exec_work() {
        // [SEQ.START]
        // [DET.START]
        // SLCO transition (p:0, id:0) | work -> wait | [t < 0 or t = 4; ldir := 1 - ldir].
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_work_0()) {
            return;
        }
        // SLCO transition (p:0, id:1) | work -> done | t >= 0 and t < 4 and req[t] = 1.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_work_1()) {
            return;
        }
        // SLCO transition (p:0, id:2) | work -> work | [t >= 0 and t < 4 and req[t] = 0; t := t + (2 * ldir) - 1].
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_work_2()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(ldir, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.req[*], 1);
    context Perm(c.t, 1);
    context Perm(c.p, 1);
    context Perm(c.v, 1);
    @*/
    // Attempt to fire a transition starting in state done.
    private void exec_done() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | done -> wait | true | v := 1.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_done_0()) {
            return;
        }
        // [SEQ.END]
    }
}

// <<< STATE_MACHINE.END (controller)

// << CLASS.END (GlobalClass)

// < MODEL.END (Elevator)