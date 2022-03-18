// > MODEL.START (Elevator)

// >> CLASS.START (GlobalClass)

// VerCors verification instructions for SLCO class GlobalClass.
class GlobalClass {
    // Class variables.
    private final int[] req;
    private volatile int t;
    private volatile int p;
    private volatile int v;

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
    GlobalClass(int[] req, int t, int p, int v) {
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
    // A list of lock requests. A value of 1 denotes that the given target is locked, and 0 implies no lock.
    private final int[] lock_requests;

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
        // Instantiate the lock requests array.
        lock_requests = new int[0];
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
    ensures \result == (v > 0);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_idle_0_s_0_n_0() {
        return v > 0;
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures \result == \old(v > 0);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | idle -> mov | v > 0.
    private boolean execute_transition_idle_0() {
        // SLCO expression | v > 0.
        if(!(t_idle_0_s_0_n_0())) {
            return false;
        }

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
    ensures \result == (t == p);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_mov_0_s_0_n_0() {
        return t == p;
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures \result == \old(t == p);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | mov -> open | t = p.
    private boolean execute_transition_mov_0() {
        // SLCO expression | t = p.
        if(!(t_mov_0_s_0_n_0())) {
            return false;
        }

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
    ensures \result == (t < p);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_mov_1_s_0_n_0() {
        return t < p;
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures \result == \old(t < p);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:1) | mov -> mov | [t < p; p := p - 1].
    private boolean execute_transition_mov_1() {
        // SLCO composite | [t < p; p := p - 1].
        // SLCO expression | t < p.
        if(!(t_mov_1_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | p := p - 1.
        range_check_assumption_t_1_s_2();
        p = p - 1;

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
    ensures \result == (t > p);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_mov_2_s_0_n_0() {
        return t > p;
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures \result == \old(t > p);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:2) | mov -> mov | [t > p; p := p + 1].
    private boolean execute_transition_mov_2() {
        // SLCO composite | [t > p; p := p + 1].
        // SLCO expression | t > p.
        if(!(t_mov_2_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | p := p + 1.
        range_check_assumption_t_2_s_2();
        p = p + 1;

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
    ensures 0 <= p && p < 4;

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
    @*/
    private void range_check_assumption_t_0_s_3() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= p && p < 4;
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures \result == \old(true);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | open -> idle | true | [true; req[p] := 0; v := 0].
    private boolean execute_transition_open_0() {
        // (Superfluous) SLCO expression | true.

        // SLCO composite | [req[p] := 0; v := 0] -> [true; req[p] := 0; v := 0].
        // (Superfluous) SLCO expression | true.
        // SLCO assignment | req[p] := 0.
        range_check_assumption_t_0_s_3();
        req[p] = (0) & 0xff;
        // SLCO assignment | v := 0.
        range_check_assumption_t_0_s_4();
        v = (0) & 0xff;

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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
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
    // A list of lock requests. A value of 1 denotes that the given target is locked, and 0 implies no lock.
    private final int[] lock_requests;

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
        // Instantiate the lock requests array.
        lock_requests = new int[0];
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
    ensures \result == (req[0] == 0);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_read_0_s_0_n_0() {
        return req[0] == 0;
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
    @*/
    private void range_check_assumption_t_0() {
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
    requires 0 <= 0 && 0 < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(req[0] == 0);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | read -> read | [req[0] = 0; req[0] := 1].
    private boolean execute_transition_read_0() {
        // SLCO composite | [req[0] = 0; req[0] := 1].
        // SLCO expression | req[0] = 0.
        if(!(t_read_0_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | req[0] := 1.
        range_check_assumption_t_0_s_2();
        req[0] = (1) & 0xff;

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
    ensures \result == (req[1] == 0);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_read_1_s_0_n_0() {
        return req[1] == 0;
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
    @*/
    private void range_check_assumption_t_1() {
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
    requires 0 <= 1 && 1 < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(req[1] == 0);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:1) | read -> read | [req[1] = 0; req[1] := 1].
    private boolean execute_transition_read_1() {
        // SLCO composite | [req[1] = 0; req[1] := 1].
        // SLCO expression | req[1] = 0.
        if(!(t_read_1_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | req[1] := 1.
        range_check_assumption_t_1_s_2();
        req[1] = (1) & 0xff;

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
    ensures \result == (req[2] == 0);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_read_2_s_0_n_0() {
        return req[2] == 0;
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
    @*/
    private void range_check_assumption_t_2() {
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
    requires 0 <= 2 && 2 < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(req[2] == 0);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:2) | read -> read | [req[2] = 0; req[2] := 1].
    private boolean execute_transition_read_2() {
        // SLCO composite | [req[2] = 0; req[2] := 1].
        // SLCO expression | req[2] = 0.
        if(!(t_read_2_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | req[2] := 1.
        range_check_assumption_t_2_s_2();
        req[2] = (1) & 0xff;

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
    ensures \result == (req[3] == 0);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_read_3_s_0_n_0() {
        return req[3] == 0;
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
    @*/
    private void range_check_assumption_t_3() {
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
    requires 0 <= 3 && 3 < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(req[3] == 0);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:3) | read -> read | [req[3] = 0; req[3] := 1].
    private boolean execute_transition_read_3() {
        // SLCO composite | [req[3] = 0; req[3] := 1].
        // SLCO expression | req[3] = 0.
        if(!(t_read_3_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | req[3] := 1.
        range_check_assumption_t_3_s_2();
        req[3] = (1) & 0xff;

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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
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
    private int ldir;
    // A list of lock requests. A value of 1 denotes that the given target is locked, and 0 implies no lock.
    private final int[] lock_requests;

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
        // Instantiate the lock requests array.
        lock_requests = new int[0];
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
    ensures \result == (v == 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_wait_0_s_0_n_0() {
        return v == 0;
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures \result == \old(v == 0);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | wait -> work | [v = 0; t := t + (2 * ldir) - 1].
    private boolean execute_transition_wait_0() {
        // SLCO composite | [v = 0; t := t + (2 * ldir) - 1].
        // SLCO expression | v = 0.
        if(!(t_wait_0_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | t := t + (2 * ldir) - 1.
        range_check_assumption_t_0_s_2();
        t = t + (2 * ldir) - 1;

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
    ensures \result == (t < 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_0_s_0_n_0() {
        return t < 0;
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
    ensures \result == (t == 4);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_0_s_0_n_1() {
        return t == 4;
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
    ensures \result == (t < 0 || t == 4);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_0_s_0_n_2() {
        if(t_work_0_s_0_n_0()) {
            // Short-circuit fix trigger.
            return true;
        }
        if(t_work_0_s_0_n_1()) {
            // Short-circuit fix trigger.
            return true;
        }
        return false;
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures \result == \old(t < 0 || t == 4);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | work -> wait | [t < 0 or t = 4; ldir := 1 - ldir].
    private boolean execute_transition_work_0() {
        // SLCO composite | [t < 0 or t = 4; ldir := 1 - ldir].
        // SLCO expression | t < 0 or t = 4.
        if(!(t_work_0_s_0_n_2())) {
            return false;
        }
        // SLCO assignment | ldir := 1 - ldir.
        range_check_assumption_t_0_s_2();
        ldir = (1 - ldir) & 0xff;

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
    ensures \result == (t >= 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_1_s_0_n_0() {
        return t >= 0;
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
    ensures \result == (t < 4);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_1_s_0_n_1() {
        return t < 4;
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
    ensures \result == (t >= 0 && t < 4);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_1_s_0_n_2() {
        if(t_work_1_s_0_n_0()) {
            if(t_work_1_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
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
    context 0 <= t && t < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (req[t] == 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_1_s_0_n_3() {
        return req[t] == 1;
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
    context 0 <= t && t < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (t >= 0 && t < 4 && req[t] == 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_1_s_0_n_4() {
        if(t_work_1_s_0_n_2()) {
            if(t_work_1_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
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
    ensures 0 <= t && t < 4;

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= t && t < 4;
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
    requires 0 <= t && t < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(t >= 0 && t < 4 && req[t] == 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:1) | work -> done | t >= 0 and t < 4 and req[t] = 1.
    private boolean execute_transition_work_1() {
        // SLCO expression | t >= 0 and t < 4 and req[t] = 1.
        if(!(t_work_1_s_0_n_4())) {
            return false;
        }

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
    ensures \result == (t >= 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_2_s_0_n_0() {
        return t >= 0;
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
    ensures \result == (t < 4);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_2_s_0_n_1() {
        return t < 4;
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
    ensures \result == (t >= 0 && t < 4);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_2_s_0_n_2() {
        if(t_work_2_s_0_n_0()) {
            if(t_work_2_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
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
    context 0 <= t && t < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (req[t] == 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_2_s_0_n_3() {
        return req[t] == 0;
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
    context 0 <= t && t < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (t >= 0 && t < 4 && req[t] == 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_2_s_0_n_4() {
        if(t_work_2_s_0_n_2()) {
            if(t_work_2_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures 0 <= t && t < 4;

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
    @*/
    private void range_check_assumption_t_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= t && t < 4;
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
    requires 0 <= t && t < 4;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(t >= 0 && t < 4 && req[t] == 0);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:2) | work -> work | [t >= 0 and t < 4 and req[t] = 0; t := t + (2 * ldir) - 1].
    private boolean execute_transition_work_2() {
        // SLCO composite | [t >= 0 and t < 4 and req[t] = 0; t := t + (2 * ldir) - 1].
        // SLCO expression | t >= 0 and t < 4 and req[t] = 0.
        if(!(t_work_2_s_0_n_4())) {
            return false;
        }
        // SLCO assignment | t := t + (2 * ldir) - 1.
        range_check_assumption_t_2_s_2();
        t = t + (2 * ldir) - 1;

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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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
    ensures (\forall* int _i; 0 <= _i && _i < req.length; req[_i] == \old(req[_i]));
    ensures t == \old(t);
    ensures p == \old(p);
    ensures v == \old(v);
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | done -> wait | true | v := 1.
    private boolean execute_transition_done_0() {
        // (Superfluous) SLCO expression | true.

        // SLCO assignment | [v := 1] -> v := 1.
        range_check_assumption_t_0_s_2();
        v = (1) & 0xff;

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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 0;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
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