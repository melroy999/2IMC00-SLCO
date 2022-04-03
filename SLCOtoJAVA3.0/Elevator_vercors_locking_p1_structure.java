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
        lock_requests = new int[3];
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 3;

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
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.v
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.v.
        if(c.v > 0) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.v
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.v
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 3;

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
    ensures \result == (c.t == c.p);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 3;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: p, 1: t]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_mov_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.p
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.t
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.p.
        //@ assert lock_requests[1] == 1; // Check c.t.
        if(c.t == c.p) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.p
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            lock_requests[1] = lock_requests[1] - 1; // Release c.t
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            return true;
        }
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 3;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: p, 1: t]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
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
    ensures \result == (c.t < c.p);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 3;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: p, 1: t]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: p]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: p, 1: t]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_mov_1_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.p.
        //@ assert lock_requests[1] == 1; // Check c.t.
        if(c.t < c.p) {
            lock_requests[1] = lock_requests[1] - 1; // Release c.t
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            return true;
        }
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 3;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: p, 1: t]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: p, 1: t]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:1) | mov -> mov | [t < p; p := p - 1].
    private boolean execute_transition_mov_1() {
        // SLCO composite | [t < p; p := p - 1].
        // SLCO expression | t < p.
        if(!(t_mov_1_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | p := p - 1.
        //@ assert lock_requests[0] == 1; // Check c.p.
        range_check_assumption_t_1_s_2();
        c.p = c.p - 1;
        lock_requests[0] = lock_requests[0] - 1; // Release c.p
        //@ assert lock_requests[0] == 0; // Verify lock activity.

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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 3;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: p, 1: t]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: p]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_mov_2_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.p.
        //@ assert lock_requests[1] == 1; // Check c.t.
        if(c.t > c.p) {
            lock_requests[1] = lock_requests[1] - 1; // Release c.t
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.p
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.t
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 3;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: p, 1: t]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

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
        //@ assert lock_requests[0] == 1; // Check c.p.
        range_check_assumption_t_2_s_2();
        c.p = c.p + 1;
        lock_requests[0] = lock_requests[0] - 1; // Release c.p
        //@ assert lock_requests[0] == 0; // Verify lock activity.

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
    context lock_requests != null && lock_requests.length == 3;

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
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.p
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.v
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        lock_requests[2] = lock_requests[2] + 1; // Acquire c.req[c.p]
        //@ assert lock_requests[2] == 1; // Verify lock activity.

        //@ assert lock_requests[0] == 1; // Check c.p.
        //@ assert lock_requests[2] == 1; // Check c.req[c.p].
        range_check_assumption_t_0_s_3();
        c.req[c.p] = (0) & 0xff;
        lock_requests[0] = lock_requests[0] - 1; // Release c.p
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.req[c.p]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        // SLCO assignment | v := 0.
        //@ assert lock_requests[1] == 1; // Check c.v.
        range_check_assumption_t_0_s_4();
        c.v = (0) & 0xff;
        lock_requests[1] = lock_requests[1] - 1; // Release c.v
        //@ assert lock_requests[1] == 0; // Verify lock activity.

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
    context lock_requests != null && lock_requests.length == 3;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state idle.
    private void exec_idle() {
        // [N_DET.START]
        // SLCO transition (p:0, id:0) | idle -> mov | v > 0.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_idle_0()) {
            return;
        }
        // [N_DET.END]
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
    context lock_requests != null && lock_requests.length == 3;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state mov.
    private void exec_mov() {
        // [N_DET.START]
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
        // [N_DET.END]
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
    context lock_requests != null && lock_requests.length == 3;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state open.
    private void exec_open() {
        // [N_DET.START]
        // SLCO transition (p:0, id:0) | open -> idle | true | [true; req[p] := 0; v := 0].
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_open_0()) {
            return;
        }
        // [N_DET.END]
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
        lock_requests = new int[4];
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 4;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: req[0]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_read_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.req[0]
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.req[0].
        if(c.req[0] == 0) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.req[0]
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 4;

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
        //@ assert lock_requests[0] == 1; // Check c.req[0].
        range_check_assumption_t_0_s_2();
        c.req[0] = (1) & 0xff;
        lock_requests[0] = lock_requests[0] - 1; // Release c.req[0]
        //@ assert lock_requests[0] == 0; // Verify lock activity.

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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 4;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [1: req[1]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_read_1_s_0_n_0() {
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.req[1]
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        //@ assert lock_requests[1] == 1; // Check c.req[1].
        if(c.req[1] == 0) {
            return true;
        }
        lock_requests[1] = lock_requests[1] - 1; // Release c.req[1]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 4;

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
        //@ assert lock_requests[1] == 1; // Check c.req[1].
        range_check_assumption_t_1_s_2();
        c.req[1] = (1) & 0xff;
        lock_requests[1] = lock_requests[1] - 1; // Release c.req[1]
        //@ assert lock_requests[1] == 0; // Verify lock activity.

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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 4;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [2: req[2]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 2) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_read_2_s_0_n_0() {
        lock_requests[2] = lock_requests[2] + 1; // Acquire c.req[2]
        //@ assert lock_requests[2] == 1; // Verify lock activity.
        //@ assert lock_requests[2] == 1; // Check c.req[2].
        if(c.req[2] == 0) {
            return true;
        }
        lock_requests[2] = lock_requests[2] - 1; // Release c.req[2]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 4;

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
        //@ assert lock_requests[2] == 1; // Check c.req[2].
        range_check_assumption_t_2_s_2();
        c.req[2] = (1) & 0xff;
        lock_requests[2] = lock_requests[2] - 1; // Release c.req[2]
        //@ assert lock_requests[2] == 0; // Verify lock activity.

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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 4;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [3: req[3]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 3) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_read_3_s_0_n_0() {
        lock_requests[3] = lock_requests[3] + 1; // Acquire c.req[3]
        //@ assert lock_requests[3] == 1; // Verify lock activity.
        //@ assert lock_requests[3] == 1; // Check c.req[3].
        if(c.req[3] == 0) {
            return true;
        }
        lock_requests[3] = lock_requests[3] - 1; // Release c.req[3]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 4;

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
        //@ assert lock_requests[3] == 1; // Check c.req[3].
        range_check_assumption_t_3_s_2();
        c.req[3] = (1) & 0xff;
        lock_requests[3] = lock_requests[3] - 1; // Release c.req[3]
        //@ assert lock_requests[3] == 0; // Verify lock activity.

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
    context lock_requests != null && lock_requests.length == 4;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state read.
    private void exec_read() {
        // [N_DET.START]
        switch(random.nextInt(4)) {
            case 0 -> {
                // SLCO transition (p:0, id:0) | read -> read | [req[0] = 0; req[0] := 1].
                //@ ghost range_check_assumption_t_3();
                if(execute_transition_read_0()) {
                    return;
                }
            }
            case 1 -> {
                // SLCO transition (p:0, id:1) | read -> read | [req[1] = 0; req[1] := 1].
                //@ ghost range_check_assumption_t_3();
                if(execute_transition_read_1()) {
                    return;
                }
            }
            case 2 -> {
                // SLCO transition (p:0, id:2) | read -> read | [req[2] = 0; req[2] := 1].
                //@ ghost range_check_assumption_t_3();
                if(execute_transition_read_2()) {
                    return;
                }
            }
            case 3 -> {
                // SLCO transition (p:0, id:3) | read -> read | [req[3] = 0; req[3] := 1].
                //@ ghost range_check_assumption_t_3();
                if(execute_transition_read_3()) {
                    return;
                }
            }
        }
        // [N_DET.END]
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
        lock_requests = new int[6];
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

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
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.v
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.v.
        if(c.v == 0) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.v
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.v
        //@ assert lock_requests[0] == 0; // Verify lock activity.
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

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
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.t
        //@ assert lock_requests[1] == 1; // Verify lock activity.

        //@ assert lock_requests[1] == 1; // Check c.t.
        range_check_assumption_t_0_s_2();
        c.t = c.t + (2 * ldir) - 1;
        lock_requests[1] = lock_requests[1] - 1; // Release c.t
        //@ assert lock_requests[1] == 0; // Verify lock activity.

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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: t]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_work_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.t
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.t.
        if(c.t < 0) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.t
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            return true;
        }
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: t]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: t]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_work_0_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.t.
        if(c.t == 4) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.t
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            return true;
        }
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: t]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: t]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
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
    ensures \result == (c.t >= 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: t]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: t]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: t]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_work_1_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.t.
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: t]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: t]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: t]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_work_1_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.t.
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: t]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: t]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: t, 1: req[2], 2: req[0], 3: req[3], 4: req[1]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_work_1_s_0_n_2() {
        if(t_work_1_s_0_n_0()) {
            if(t_work_1_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.req[2]
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        lock_requests[2] = lock_requests[2] + 1; // Acquire c.req[0]
        //@ assert lock_requests[2] == 1; // Verify lock activity.
        lock_requests[3] = lock_requests[3] + 1; // Acquire c.req[3]
        //@ assert lock_requests[3] == 1; // Verify lock activity.
        lock_requests[4] = lock_requests[4] + 1; // Acquire c.req[1]
        //@ assert lock_requests[4] == 1; // Verify lock activity.
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: t]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: t, 1: req[2], 2: req[0], 3: req[3], 4: req[1]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_work_1_s_0_n_3() {
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.req[2]
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        lock_requests[2] = lock_requests[2] + 1; // Acquire c.req[0]
        //@ assert lock_requests[2] == 1; // Verify lock activity.
        lock_requests[3] = lock_requests[3] + 1; // Acquire c.req[3]
        //@ assert lock_requests[3] == 1; // Verify lock activity.
        lock_requests[4] = lock_requests[4] + 1; // Acquire c.req[1]
        //@ assert lock_requests[4] == 1; // Verify lock activity.
        lock_requests[5] = lock_requests[5] + 1; // Acquire c.req[c.t]
        //@ assert lock_requests[5] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.t.
        //@ assert lock_requests[5] == 1; // Check c.req[c.t].
        if(c.req[c.t] == 1) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.t
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            lock_requests[1] = lock_requests[1] - 1; // Release c.req[2]
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            lock_requests[2] = lock_requests[2] - 1; // Release c.req[0]
            //@ assert lock_requests[2] == 0; // Verify lock activity.
            lock_requests[3] = lock_requests[3] - 1; // Release c.req[3]
            //@ assert lock_requests[3] == 0; // Verify lock activity.
            lock_requests[4] = lock_requests[4] - 1; // Release c.req[1]
            //@ assert lock_requests[4] == 0; // Verify lock activity.
            lock_requests[5] = lock_requests[5] - 1; // Release c.req[c.t]
            //@ assert lock_requests[5] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[5] = lock_requests[5] - 1; // Release c.req[c.t]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: t]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: t, 1: req[2], 2: req[0], 3: req[3], 4: req[1]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: t]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: t, 1: req[2], 2: req[0], 3: req[3], 4: req[1]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
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
    ensures \result == (c.t >= 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures ldir == \old(ldir);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.req.length; c.req[_i] == \old(c.req[_i]));
    ensures c.t == \old(c.t);
    ensures c.p == \old(c.p);
    ensures c.v == \old(c.v);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: t, 1: req[2], 2: req[0], 3: req[3], 4: req[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: t, 1: req[2], 2: req[0], 3: req[3], 4: req[1]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_2_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.t.
        if(c.t >= 0) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.t
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.req[2]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.req[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.req[3]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.req[1]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: t, 1: req[2], 2: req[0], 3: req[3], 4: req[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: t, 1: req[2], 2: req[0], 3: req[3], 4: req[1]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_2_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.t.
        if(c.t < 4) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.t
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.req[2]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.req[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.req[3]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.req[1]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: t, 1: req[2], 2: req[0], 3: req[3], 4: req[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: t, 1: req[2], 2: req[0], 3: req[3], 4: req[1]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: t, 1: req[2], 2: req[0], 3: req[3], 4: req[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: t]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_work_2_s_0_n_3() {
        //@ assert lock_requests[0] == 1; // Check c.t.
        //@ assert lock_requests[1] == 1; // Check c.req[2].
        //@ assert lock_requests[2] == 1; // Check c.req[0].
        //@ assert lock_requests[3] == 1; // Check c.req[3].
        //@ assert lock_requests[4] == 1; // Check c.req[1].
        if(c.req[c.t] == 0) {
            lock_requests[1] = lock_requests[1] - 1; // Release c.req[2]
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            lock_requests[2] = lock_requests[2] - 1; // Release c.req[0]
            //@ assert lock_requests[2] == 0; // Verify lock activity.
            lock_requests[3] = lock_requests[3] - 1; // Release c.req[3]
            //@ assert lock_requests[3] == 0; // Verify lock activity.
            lock_requests[4] = lock_requests[4] - 1; // Release c.req[1]
            //@ assert lock_requests[4] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.t
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.req[2]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.req[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.req[3]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.req[1]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        return false;
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: t, 1: req[2], 2: req[0], 3: req[3], 4: req[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: t]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: t, 1: req[2], 2: req[0], 3: req[3], 4: req[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

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
        //@ assert lock_requests[0] == 1; // Check c.t.
        range_check_assumption_t_2_s_2();
        c.t = c.t + (2 * ldir) - 1;
        lock_requests[0] = lock_requests[0] - 1; // Release c.t
        //@ assert lock_requests[0] == 0; // Verify lock activity.

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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 6;

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
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.v
        //@ assert lock_requests[0] == 1; // Verify lock activity.

        //@ assert lock_requests[0] == 1; // Check c.v.
        range_check_assumption_t_0_s_2();
        c.v = (1) & 0xff;
        lock_requests[0] = lock_requests[0] - 1; // Release c.v
        //@ assert lock_requests[0] == 0; // Verify lock activity.

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
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state wait.
    private void exec_wait() {
        // [N_DET.START]
        // SLCO transition (p:0, id:0) | wait -> work | [v = 0; t := t + (2 * ldir) - 1].
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_wait_0()) {
            return;
        }
        // [N_DET.END]
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
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state work.
    private void exec_work() {
        // [N_DET.START]
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
        // [N_DET.END]
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
    context lock_requests != null && lock_requests.length == 6;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state done.
    private void exec_done() {
        // [N_DET.START]
        // SLCO transition (p:0, id:0) | done -> wait | true | v := 1.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_done_0()) {
            return;
        }
        // [N_DET.END]
    }
}

// <<< STATE_MACHINE.END (controller)

// << CLASS.END (GlobalClass)

// < MODEL.END (Elevator)