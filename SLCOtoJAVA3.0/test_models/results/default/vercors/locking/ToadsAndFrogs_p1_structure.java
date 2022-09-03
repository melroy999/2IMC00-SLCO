// > MODEL.START (ToadsAndFrogs)

// >> CLASS.START (GlobalClass)

// VerCors verification instructions for SLCO class GlobalClass.
class GlobalClass {
    // Class variables.
    private volatile int y;
    private volatile int tmin;
    private volatile int fmax;
    private final int[] a;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.y, 1);
    ensures Perm(this.tmin, 1);
    ensures Perm(this.fmax, 1);
    ensures Perm(this.a, 1);

    // Require that the given values are not null.
    requires a != null;

    // Ensure that the right values are assigned.
    ensures this.y == y;
    ensures this.tmin == tmin;
    ensures this.fmax == fmax;
    ensures this.a == a;
    @*/
    GlobalClass(int y, int tmin, int fmax, int[] a) {
        // Instantiate the class variables.
        this.y = y;
        this.tmin = tmin;
        this.fmax = fmax;
        this.a = a;
    }
}

// >>> STATE_MACHINE.START (toad)

// VerCors verification instructions for SLCO state machine toad.
class GlobalClass_toadThread {
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
    GlobalClass_toadThread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;
        // Instantiate the lock requests array.
        lock_requests = new int[13];
    }

    // SLCO expression wrapper | y > 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 0);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.y
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.y.
        if(c.y > 0) {
            return true;
        }
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.tmin
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | tmin != y - 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin != c.y - 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_0_s_0_n_1() {
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.tmin
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.tmin.
        return c.tmin != c.y - 1;
    }

    // SLCO expression wrapper | y > 0 and tmin != y - 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 0 && c.tmin != c.y - 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_0_s_0_n_2() {
        if(t_q_0_s_0_n_0()) {
            if(t_q_0_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        lock_requests[2] = lock_requests[2] + 1; // Acquire c.a[0]
        //@ assert lock_requests[2] == 1; // Verify lock activity.
        lock_requests[3] = lock_requests[3] + 1; // Acquire c.a[1]
        //@ assert lock_requests[3] == 1; // Verify lock activity.
        lock_requests[4] = lock_requests[4] + 1; // Acquire c.a[2]
        //@ assert lock_requests[4] == 1; // Verify lock activity.
        lock_requests[5] = lock_requests[5] + 1; // Acquire c.a[3]
        //@ assert lock_requests[5] == 1; // Verify lock activity.
        lock_requests[6] = lock_requests[6] + 1; // Acquire c.a[4]
        //@ assert lock_requests[6] == 1; // Verify lock activity.
        lock_requests[7] = lock_requests[7] + 1; // Acquire c.a[5]
        //@ assert lock_requests[7] == 1; // Verify lock activity.
        lock_requests[8] = lock_requests[8] + 1; // Acquire c.a[6]
        //@ assert lock_requests[8] == 1; // Verify lock activity.
        lock_requests[9] = lock_requests[9] + 1; // Acquire c.a[7]
        //@ assert lock_requests[9] == 1; // Verify lock activity.
        lock_requests[10] = lock_requests[10] + 1; // Acquire c.a[8]
        //@ assert lock_requests[10] == 1; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | a[y - 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 11: a[y], 12: a[(y - 1)]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 11 || _i == 12) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_0_s_0_n_3() {
        lock_requests[2] = lock_requests[2] + 1; // Acquire c.a[0]
        //@ assert lock_requests[2] == 1; // Verify lock activity.
        lock_requests[3] = lock_requests[3] + 1; // Acquire c.a[1]
        //@ assert lock_requests[3] == 1; // Verify lock activity.
        lock_requests[4] = lock_requests[4] + 1; // Acquire c.a[2]
        //@ assert lock_requests[4] == 1; // Verify lock activity.
        lock_requests[5] = lock_requests[5] + 1; // Acquire c.a[3]
        //@ assert lock_requests[5] == 1; // Verify lock activity.
        lock_requests[6] = lock_requests[6] + 1; // Acquire c.a[4]
        //@ assert lock_requests[6] == 1; // Verify lock activity.
        lock_requests[7] = lock_requests[7] + 1; // Acquire c.a[5]
        //@ assert lock_requests[7] == 1; // Verify lock activity.
        lock_requests[8] = lock_requests[8] + 1; // Acquire c.a[6]
        //@ assert lock_requests[8] == 1; // Verify lock activity.
        lock_requests[9] = lock_requests[9] + 1; // Acquire c.a[7]
        //@ assert lock_requests[9] == 1; // Verify lock activity.
        lock_requests[10] = lock_requests[10] + 1; // Acquire c.a[8]
        //@ assert lock_requests[10] == 1; // Verify lock activity.
        lock_requests[11] = lock_requests[11] + 1; // Acquire c.a[c.y]
        //@ assert lock_requests[11] == 1; // Verify lock activity.
        lock_requests[12] = lock_requests[12] + 1; // Acquire c.a[(c.y - 1)]
        //@ assert lock_requests[12] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[12] == 1; // Check c.a[(c.y - 1)].
        if(c.a[c.y - 1] == 1) {
            lock_requests[1] = lock_requests[1] - 1; // Release c.tmin
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
            //@ assert lock_requests[2] == 0; // Verify lock activity.
            lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
            //@ assert lock_requests[3] == 0; // Verify lock activity.
            lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
            //@ assert lock_requests[4] == 0; // Verify lock activity.
            lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
            //@ assert lock_requests[5] == 0; // Verify lock activity.
            lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
            //@ assert lock_requests[6] == 0; // Verify lock activity.
            lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
            //@ assert lock_requests[7] == 0; // Verify lock activity.
            lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
            //@ assert lock_requests[8] == 0; // Verify lock activity.
            lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
            //@ assert lock_requests[9] == 0; // Verify lock activity.
            lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
            //@ assert lock_requests[10] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[11] = lock_requests[11] - 1; // Release c.a[c.y]
        //@ assert lock_requests[11] == 0; // Verify lock activity.
        lock_requests[12] = lock_requests[12] - 1; // Release c.a[(c.y - 1)]
        //@ assert lock_requests[12] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y > 0 and tmin != y - 1 and a[y - 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 0 && c.tmin != c.y - 1 && c.a[c.y - 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 11: a[y], 12: a[(y - 1)]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 11 || _i == 12) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_0_s_0_n_4() {
        if(t_q_0_s_0_n_2()) {
            if(t_q_0_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_4() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y > 0 && c.tmin != c.y - 1 && c.a[c.y - 1] == 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | q -> q | [y > 0 and tmin != y - 1 and a[y - 1] = 1; a[y] := 1; y := y - 1; a[y] := 0].
    private boolean execute_transition_q_0() {
        // SLCO composite | [y > 0 and tmin != y - 1 and a[y - 1] = 1; a[y] := 1; y := y - 1; a[y] := 0].
        // SLCO expression | y > 0 and tmin != y - 1 and a[y - 1] = 1.
        if(!(t_q_0_s_0_n_4())) {
            return false;
        }
        // SLCO assignment | a[y] := 1.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[11] == 1; // Check c.a[c.y].
        range_check_assumption_t_0_s_2();
        c.a[c.y] = 1;
        lock_requests[11] = lock_requests[11] - 1; // Release c.a[c.y]
        //@ assert lock_requests[11] == 0; // Verify lock activity.
        // SLCO assignment | y := y - 1.
        //@ assert lock_requests[0] == 1; // Check c.y.
        range_check_assumption_t_0_s_3();
        c.y = c.y - 1;
        // SLCO assignment | a[y] := 0.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[12] == 1; // Check c.a[(c.y - 1)].
        range_check_assumption_t_0_s_4();
        c.a[c.y] = 0;
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[12] = lock_requests[12] - 1; // Release c.a[(c.y - 1)]
        //@ assert lock_requests[12] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | y > 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 0);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_1_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        return c.y > 0;
    }

    // SLCO expression wrapper | tmin = y - 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin == c.y - 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_1_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.tmin.
        return c.tmin == c.y - 1;
    }

    // SLCO expression wrapper | y > 0 and tmin = y - 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 0 && c.tmin == c.y - 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_1_s_0_n_2() {
        if(t_q_1_s_0_n_0()) {
            if(t_q_1_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y - 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_1_s_0_n_3() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        return c.a[c.y - 1] == 1;
    }

    // SLCO expression wrapper | y > 0 and tmin = y - 1 and a[y - 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 0 && c.tmin == c.y - 1 && c.a[c.y - 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_1_s_0_n_4() {
        if(t_q_1_s_0_n_2()) {
            if(t_q_1_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_5() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y > 0 && c.tmin == c.y - 1 && c.a[c.y - 1] == 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:1) | q -> q | [y > 0 and tmin = y - 1 and a[y - 1] = 1; a[y] := 1; tmin := y; y := y - 1; a[y] := 0].
    private boolean execute_transition_q_1() {
        // SLCO composite | [y > 0 and tmin = y - 1 and a[y - 1] = 1; a[y] := 1; tmin := y; y := y - 1; a[y] := 0].
        // SLCO expression | y > 0 and tmin = y - 1 and a[y - 1] = 1.
        if(!(t_q_1_s_0_n_4())) {
            return false;
        }
        // SLCO assignment | a[y] := 1.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        range_check_assumption_t_1_s_2();
        c.a[c.y] = 1;
        // SLCO assignment | tmin := y.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.tmin.
        range_check_assumption_t_1_s_3();
        c.tmin = c.y;
        lock_requests[1] = lock_requests[1] - 1; // Release c.tmin
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        // SLCO assignment | y := y - 1.
        //@ assert lock_requests[0] == 1; // Check c.y.
        range_check_assumption_t_1_s_4();
        c.y = c.y - 1;
        // SLCO assignment | a[y] := 0.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        range_check_assumption_t_1_s_5();
        c.a[c.y] = 0;
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | y > 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        return c.y > 1;
    }

    // SLCO expression wrapper | tmin != y - 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin != c.y - 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.tmin.
        return c.tmin != c.y - 2;
    }

    // SLCO expression wrapper | y > 1 and tmin != y - 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.tmin != c.y - 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_2() {
        if(t_q_2_s_0_n_0()) {
            if(t_q_2_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y - 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_3() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        return c.a[c.y - 2] == 1;
    }

    // SLCO expression wrapper | y > 1 and tmin != y - 2 and a[y - 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.tmin != c.y - 2 && c.a[c.y - 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_4() {
        if(t_q_2_s_0_n_2()) {
            if(t_q_2_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_5() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        if(c.a[c.y - 1] == 2) {
            lock_requests[1] = lock_requests[1] - 1; // Release c.tmin
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            return true;
        }
        return false;
    }

    // SLCO expression wrapper | y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.tmin != c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_6() {
        if(t_q_2_s_0_n_4()) {
            if(t_q_2_s_0_n_5()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2_s_4() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 2 && c.y - 2 < 9;
    ensures 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 2 && c.y - 2 < 9;
    requires 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y > 1 && c.tmin != c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:2) | q -> q | [y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; y := y - 2; a[y] := 0].
    private boolean execute_transition_q_2() {
        // SLCO composite | [y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; y := y - 2; a[y] := 0].
        // SLCO expression | y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
        if(!(t_q_2_s_0_n_6())) {
            return false;
        }
        // SLCO assignment | a[y] := 1.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        range_check_assumption_t_2_s_2();
        c.a[c.y] = 1;
        // SLCO assignment | y := y - 2.
        //@ assert lock_requests[0] == 1; // Check c.y.
        range_check_assumption_t_2_s_3();
        c.y = c.y - 2;
        // SLCO assignment | a[y] := 0.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        range_check_assumption_t_2_s_4();
        c.a[c.y] = 0;
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | y > 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        if(c.y > 1) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.tmin
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | tmin = y - 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin == c.y - 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.tmin.
        if(c.tmin == c.y - 2) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.tmin
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y > 1 and tmin = y - 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.tmin == c.y - 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_2() {
        if(t_q_3_s_0_n_0()) {
            if(t_q_3_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y - 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_3() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        if(c.a[c.y - 2] == 1) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.tmin
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y > 1 and tmin = y - 2 and a[y - 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.tmin == c.y - 2 && c.a[c.y - 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_4() {
        if(t_q_3_s_0_n_2()) {
            if(t_q_3_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_5() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        if(c.a[c.y - 1] == 2) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.tmin
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.tmin == c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_6() {
        if(t_q_3_s_0_n_4()) {
            if(t_q_3_s_0_n_5()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_5() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 2 && c.y - 2 < 9;
    ensures 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 2 && c.y - 2 < 9;
    requires 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y > 1 && c.tmin == c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:3) | q -> q | [y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; tmin := y; y := y - 2; a[y] := 0].
    private boolean execute_transition_q_3() {
        // SLCO composite | [y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; tmin := y; y := y - 2; a[y] := 0].
        // SLCO expression | y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
        if(!(t_q_3_s_0_n_6())) {
            return false;
        }
        // SLCO assignment | a[y] := 1.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        range_check_assumption_t_3_s_2();
        c.a[c.y] = 1;
        // SLCO assignment | tmin := y.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.tmin.
        range_check_assumption_t_3_s_3();
        c.tmin = c.y;
        lock_requests[1] = lock_requests[1] - 1; // Release c.tmin
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        // SLCO assignment | y := y - 2.
        //@ assert lock_requests[0] == 1; // Check c.y.
        range_check_assumption_t_3_s_4();
        c.y = c.y - 2;
        // SLCO assignment | a[y] := 0.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        range_check_assumption_t_3_s_5();
        c.a[c.y] = 0;
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state q.
    private void exec_q() {
        // [SEQ.START]
        // [DET.START]
        // SLCO transition (p:0, id:0) | q -> q | [y > 0 and tmin != y - 1 and a[y - 1] = 1; a[y] := 1; y := y - 1; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_0()) {
            return;
        }
        // SLCO transition (p:0, id:1) | q -> q | [y > 0 and tmin = y - 1 and a[y - 1] = 1; a[y] := 1; tmin := y; y := y - 1; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_1()) {
            return;
        }
        // SLCO transition (p:0, id:2) | q -> q | [y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; y := y - 2; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_2()) {
            return;
        }
        // SLCO transition (p:0, id:3) | q -> q | [y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; tmin := y; y := y - 2; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_3()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }
}

// <<< STATE_MACHINE.END (toad)

// >>> STATE_MACHINE.START (frog)

// VerCors verification instructions for SLCO state machine frog.
class GlobalClass_frogThread {
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
    GlobalClass_frogThread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;
        // Instantiate the lock requests array.
        lock_requests = new int[13];
    }

    // SLCO expression wrapper | y < 8.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 8);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.y
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.y.
        if(c.y < 8) {
            return true;
        }
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.fmax
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | fmax != y + 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.fmax != c.y + 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_0_s_0_n_1() {
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.fmax
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.fmax.
        return c.fmax != c.y + 1;
    }

    // SLCO expression wrapper | y < 8 and fmax != y + 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 8 && c.fmax != c.y + 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_0_s_0_n_2() {
        if(t_q_0_s_0_n_0()) {
            if(t_q_0_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        lock_requests[2] = lock_requests[2] + 1; // Acquire c.a[0]
        //@ assert lock_requests[2] == 1; // Verify lock activity.
        lock_requests[3] = lock_requests[3] + 1; // Acquire c.a[1]
        //@ assert lock_requests[3] == 1; // Verify lock activity.
        lock_requests[4] = lock_requests[4] + 1; // Acquire c.a[2]
        //@ assert lock_requests[4] == 1; // Verify lock activity.
        lock_requests[5] = lock_requests[5] + 1; // Acquire c.a[3]
        //@ assert lock_requests[5] == 1; // Verify lock activity.
        lock_requests[6] = lock_requests[6] + 1; // Acquire c.a[4]
        //@ assert lock_requests[6] == 1; // Verify lock activity.
        lock_requests[7] = lock_requests[7] + 1; // Acquire c.a[5]
        //@ assert lock_requests[7] == 1; // Verify lock activity.
        lock_requests[8] = lock_requests[8] + 1; // Acquire c.a[6]
        //@ assert lock_requests[8] == 1; // Verify lock activity.
        lock_requests[9] = lock_requests[9] + 1; // Acquire c.a[7]
        //@ assert lock_requests[9] == 1; // Verify lock activity.
        lock_requests[10] = lock_requests[10] + 1; // Acquire c.a[8]
        //@ assert lock_requests[10] == 1; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | a[y + 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 11: a[y], 12: a[(y + 1)]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 11 || _i == 12) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_0_s_0_n_3() {
        lock_requests[2] = lock_requests[2] + 1; // Acquire c.a[0]
        //@ assert lock_requests[2] == 1; // Verify lock activity.
        lock_requests[3] = lock_requests[3] + 1; // Acquire c.a[1]
        //@ assert lock_requests[3] == 1; // Verify lock activity.
        lock_requests[4] = lock_requests[4] + 1; // Acquire c.a[2]
        //@ assert lock_requests[4] == 1; // Verify lock activity.
        lock_requests[5] = lock_requests[5] + 1; // Acquire c.a[3]
        //@ assert lock_requests[5] == 1; // Verify lock activity.
        lock_requests[6] = lock_requests[6] + 1; // Acquire c.a[4]
        //@ assert lock_requests[6] == 1; // Verify lock activity.
        lock_requests[7] = lock_requests[7] + 1; // Acquire c.a[5]
        //@ assert lock_requests[7] == 1; // Verify lock activity.
        lock_requests[8] = lock_requests[8] + 1; // Acquire c.a[6]
        //@ assert lock_requests[8] == 1; // Verify lock activity.
        lock_requests[9] = lock_requests[9] + 1; // Acquire c.a[7]
        //@ assert lock_requests[9] == 1; // Verify lock activity.
        lock_requests[10] = lock_requests[10] + 1; // Acquire c.a[8]
        //@ assert lock_requests[10] == 1; // Verify lock activity.
        lock_requests[11] = lock_requests[11] + 1; // Acquire c.a[c.y]
        //@ assert lock_requests[11] == 1; // Verify lock activity.
        lock_requests[12] = lock_requests[12] + 1; // Acquire c.a[(c.y + 1)]
        //@ assert lock_requests[12] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[12] == 1; // Check c.a[(c.y + 1)].
        if(c.a[c.y + 1] == 2) {
            lock_requests[1] = lock_requests[1] - 1; // Release c.fmax
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
            //@ assert lock_requests[2] == 0; // Verify lock activity.
            lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
            //@ assert lock_requests[3] == 0; // Verify lock activity.
            lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
            //@ assert lock_requests[4] == 0; // Verify lock activity.
            lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
            //@ assert lock_requests[5] == 0; // Verify lock activity.
            lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
            //@ assert lock_requests[6] == 0; // Verify lock activity.
            lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
            //@ assert lock_requests[7] == 0; // Verify lock activity.
            lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
            //@ assert lock_requests[8] == 0; // Verify lock activity.
            lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
            //@ assert lock_requests[9] == 0; // Verify lock activity.
            lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
            //@ assert lock_requests[10] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[11] = lock_requests[11] - 1; // Release c.a[c.y]
        //@ assert lock_requests[11] == 0; // Verify lock activity.
        lock_requests[12] = lock_requests[12] - 1; // Release c.a[(c.y + 1)]
        //@ assert lock_requests[12] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y < 8 and fmax != y + 1 and a[y + 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 8 && c.fmax != c.y + 1 && c.a[c.y + 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 11: a[y], 12: a[(y + 1)]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 11 || _i == 12) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_0_s_0_n_4() {
        if(t_q_0_s_0_n_2()) {
            if(t_q_0_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_4() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y < 8 && c.fmax != c.y + 1 && c.a[c.y + 1] == 2);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | q -> q | [y < 8 and fmax != y + 1 and a[y + 1] = 2; a[y] := 2; y := y + 1; a[y] := 0].
    private boolean execute_transition_q_0() {
        // SLCO composite | [y < 8 and fmax != y + 1 and a[y + 1] = 2; a[y] := 2; y := y + 1; a[y] := 0].
        // SLCO expression | y < 8 and fmax != y + 1 and a[y + 1] = 2.
        if(!(t_q_0_s_0_n_4())) {
            return false;
        }
        // SLCO assignment | a[y] := 2.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[11] == 1; // Check c.a[c.y].
        range_check_assumption_t_0_s_2();
        c.a[c.y] = 2;
        lock_requests[11] = lock_requests[11] - 1; // Release c.a[c.y]
        //@ assert lock_requests[11] == 0; // Verify lock activity.
        // SLCO assignment | y := y + 1.
        //@ assert lock_requests[0] == 1; // Check c.y.
        range_check_assumption_t_0_s_3();
        c.y = c.y + 1;
        // SLCO assignment | a[y] := 0.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[12] == 1; // Check c.a[(c.y + 1)].
        range_check_assumption_t_0_s_4();
        c.a[c.y] = 0;
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[12] = lock_requests[12] - 1; // Release c.a[(c.y + 1)]
        //@ assert lock_requests[12] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | y < 8.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 8);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_1_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        return c.y < 8;
    }

    // SLCO expression wrapper | fmax = y + 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.fmax == c.y + 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_1_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.fmax.
        return c.fmax == c.y + 1;
    }

    // SLCO expression wrapper | y < 8 and fmax = y + 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 8 && c.fmax == c.y + 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_1_s_0_n_2() {
        if(t_q_1_s_0_n_0()) {
            if(t_q_1_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y + 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_1_s_0_n_3() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        return c.a[c.y + 1] == 2;
    }

    // SLCO expression wrapper | y < 8 and fmax = y + 1 and a[y + 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 8 && c.fmax == c.y + 1 && c.a[c.y + 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_1_s_0_n_4() {
        if(t_q_1_s_0_n_2()) {
            if(t_q_1_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_5() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y < 8 && c.fmax == c.y + 1 && c.a[c.y + 1] == 2);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:1) | q -> q | [y < 8 and fmax = y + 1 and a[y + 1] = 2; a[y] := 2; fmax := y; y := y + 1; a[y] := 0].
    private boolean execute_transition_q_1() {
        // SLCO composite | [y < 8 and fmax = y + 1 and a[y + 1] = 2; a[y] := 2; fmax := y; y := y + 1; a[y] := 0].
        // SLCO expression | y < 8 and fmax = y + 1 and a[y + 1] = 2.
        if(!(t_q_1_s_0_n_4())) {
            return false;
        }
        // SLCO assignment | a[y] := 2.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        range_check_assumption_t_1_s_2();
        c.a[c.y] = 2;
        // SLCO assignment | fmax := y.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.fmax.
        range_check_assumption_t_1_s_3();
        c.fmax = c.y;
        lock_requests[1] = lock_requests[1] - 1; // Release c.fmax
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        // SLCO assignment | y := y + 1.
        //@ assert lock_requests[0] == 1; // Check c.y.
        range_check_assumption_t_1_s_4();
        c.y = c.y + 1;
        // SLCO assignment | a[y] := 0.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        range_check_assumption_t_1_s_5();
        c.a[c.y] = 0;
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | y < 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        return c.y < 7;
    }

    // SLCO expression wrapper | fmax != y + 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.fmax != c.y + 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.fmax.
        return c.fmax != c.y + 2;
    }

    // SLCO expression wrapper | y < 7 and fmax != y + 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7 && c.fmax != c.y + 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_2() {
        if(t_q_2_s_0_n_0()) {
            if(t_q_2_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_3() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        return c.a[c.y + 1] == 1;
    }

    // SLCO expression wrapper | y < 7 and fmax != y + 2 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7 && c.fmax != c.y + 2 && c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_4() {
        if(t_q_2_s_0_n_2()) {
            if(t_q_2_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y + 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_5() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        if(c.a[c.y + 2] == 2) {
            lock_requests[1] = lock_requests[1] - 1; // Release c.fmax
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            return true;
        }
        return false;
    }

    // SLCO expression wrapper | y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7 && c.fmax != c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_q_2_s_0_n_6() {
        if(t_q_2_s_0_n_4()) {
            if(t_q_2_s_0_n_5()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2_s_4() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y + 1 && c.y + 1 < 9;
    ensures 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y + 1 && c.y + 1 < 9;
    requires 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y < 7 && c.fmax != c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:2) | q -> q | [y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; y := y + 2; a[y] := 0].
    private boolean execute_transition_q_2() {
        // SLCO composite | [y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; y := y + 2; a[y] := 0].
        // SLCO expression | y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
        if(!(t_q_2_s_0_n_6())) {
            return false;
        }
        // SLCO assignment | a[y] := 2.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        range_check_assumption_t_2_s_2();
        c.a[c.y] = 2;
        // SLCO assignment | y := y + 2.
        //@ assert lock_requests[0] == 1; // Check c.y.
        range_check_assumption_t_2_s_3();
        c.y = c.y + 2;
        // SLCO assignment | a[y] := 0.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        range_check_assumption_t_2_s_4();
        c.a[c.y] = 0;
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | y < 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        if(c.y < 7) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.fmax
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | fmax = y + 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.fmax == c.y + 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.fmax.
        if(c.fmax == c.y + 2) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.fmax
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y < 7 and fmax = y + 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7 && c.fmax == c.y + 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_2() {
        if(t_q_3_s_0_n_0()) {
            if(t_q_3_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_3() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        if(c.a[c.y + 1] == 1) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.fmax
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y < 7 and fmax = y + 2 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7 && c.fmax == c.y + 2 && c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_4() {
        if(t_q_3_s_0_n_2()) {
            if(t_q_3_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y + 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_5() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        if(c.a[c.y + 2] == 2) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.fmax
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7 && c.fmax == c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_q_3_s_0_n_6() {
        if(t_q_3_s_0_n_4()) {
            if(t_q_3_s_0_n_5()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_5() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y + 1 && c.y + 1 < 9;
    ensures 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y + 1 && c.y + 1 < 9;
    requires 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y < 7 && c.fmax == c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: fmax, 2: a[0], 3: a[1], 4: a[2], 5: a[3], 6: a[4], 7: a[5], 8: a[6], 9: a[7], 10: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:3) | q -> q | [y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; fmax := y; y := y + 2; a[y] := 0].
    private boolean execute_transition_q_3() {
        // SLCO composite | [y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; fmax := y; y := y + 2; a[y] := 0].
        // SLCO expression | y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
        if(!(t_q_3_s_0_n_6())) {
            return false;
        }
        // SLCO assignment | a[y] := 2.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        range_check_assumption_t_3_s_2();
        c.a[c.y] = 2;
        // SLCO assignment | fmax := y.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.fmax.
        range_check_assumption_t_3_s_3();
        c.fmax = c.y;
        lock_requests[1] = lock_requests[1] - 1; // Release c.fmax
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        // SLCO assignment | y := y + 2.
        //@ assert lock_requests[0] == 1; // Check c.y.
        range_check_assumption_t_3_s_4();
        c.y = c.y + 2;
        // SLCO assignment | a[y] := 0.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.a[0].
        //@ assert lock_requests[3] == 1; // Check c.a[1].
        //@ assert lock_requests[4] == 1; // Check c.a[2].
        //@ assert lock_requests[5] == 1; // Check c.a[3].
        //@ assert lock_requests[6] == 1; // Check c.a[4].
        //@ assert lock_requests[7] == 1; // Check c.a[5].
        //@ assert lock_requests[8] == 1; // Check c.a[6].
        //@ assert lock_requests[9] == 1; // Check c.a[7].
        //@ assert lock_requests[10] == 1; // Check c.a[8].
        range_check_assumption_t_3_s_5();
        c.a[c.y] = 0;
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[0]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[1]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[2]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[3]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[4]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[5]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[6]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[7]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[8]
        //@ assert lock_requests[10] == 0; // Verify lock activity.

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 13;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state q.
    private void exec_q() {
        // [SEQ.START]
        // [DET.START]
        // SLCO transition (p:0, id:0) | q -> q | [y < 8 and fmax != y + 1 and a[y + 1] = 2; a[y] := 2; y := y + 1; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_0()) {
            return;
        }
        // SLCO transition (p:0, id:1) | q -> q | [y < 8 and fmax = y + 1 and a[y + 1] = 2; a[y] := 2; fmax := y; y := y + 1; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_1()) {
            return;
        }
        // SLCO transition (p:0, id:2) | q -> q | [y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; y := y + 2; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_2()) {
            return;
        }
        // SLCO transition (p:0, id:3) | q -> q | [y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; fmax := y; y := y + 2; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_3()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }
}

// <<< STATE_MACHINE.END (frog)

// >>> STATE_MACHINE.START (control)

// VerCors verification instructions for SLCO state machine control.
class GlobalClass_controlThread {
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
    GlobalClass_controlThread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;
        // Instantiate the lock requests array.
        lock_requests = new int[12];
    }

    // SLCO expression wrapper | y = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 0);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.y
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.y.
        if(c.y == 0) {
            return true;
        }
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.a[0]
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        lock_requests[2] = lock_requests[2] + 1; // Acquire c.a[1]
        //@ assert lock_requests[2] == 1; // Verify lock activity.
        lock_requests[3] = lock_requests[3] + 1; // Acquire c.a[2]
        //@ assert lock_requests[3] == 1; // Verify lock activity.
        lock_requests[4] = lock_requests[4] + 1; // Acquire c.a[3]
        //@ assert lock_requests[4] == 1; // Verify lock activity.
        lock_requests[5] = lock_requests[5] + 1; // Acquire c.a[4]
        //@ assert lock_requests[5] == 1; // Verify lock activity.
        lock_requests[6] = lock_requests[6] + 1; // Acquire c.a[5]
        //@ assert lock_requests[6] == 1; // Verify lock activity.
        lock_requests[7] = lock_requests[7] + 1; // Acquire c.a[6]
        //@ assert lock_requests[7] == 1; // Verify lock activity.
        lock_requests[8] = lock_requests[8] + 1; // Acquire c.a[7]
        //@ assert lock_requests[8] == 1; // Verify lock activity.
        lock_requests[9] = lock_requests[9] + 1; // Acquire c.a[8]
        //@ assert lock_requests[9] == 1; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8], 10: a[y + 2]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_0_s_0_n_1() {
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.a[0]
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        lock_requests[2] = lock_requests[2] + 1; // Acquire c.a[1]
        //@ assert lock_requests[2] == 1; // Verify lock activity.
        lock_requests[3] = lock_requests[3] + 1; // Acquire c.a[2]
        //@ assert lock_requests[3] == 1; // Verify lock activity.
        lock_requests[4] = lock_requests[4] + 1; // Acquire c.a[3]
        //@ assert lock_requests[4] == 1; // Verify lock activity.
        lock_requests[5] = lock_requests[5] + 1; // Acquire c.a[4]
        //@ assert lock_requests[5] == 1; // Verify lock activity.
        lock_requests[6] = lock_requests[6] + 1; // Acquire c.a[5]
        //@ assert lock_requests[6] == 1; // Verify lock activity.
        lock_requests[7] = lock_requests[7] + 1; // Acquire c.a[6]
        //@ assert lock_requests[7] == 1; // Verify lock activity.
        lock_requests[8] = lock_requests[8] + 1; // Acquire c.a[7]
        //@ assert lock_requests[8] == 1; // Verify lock activity.
        lock_requests[9] = lock_requests[9] + 1; // Acquire c.a[8]
        //@ assert lock_requests[9] == 1; // Verify lock activity.
        lock_requests[10] = lock_requests[10] + 1; // Acquire c.a[c.y + 2]
        //@ assert lock_requests[10] == 1; // Verify lock activity.
        lock_requests[11] = lock_requests[11] + 1; // Acquire c.a[c.y + 1]
        //@ assert lock_requests[11] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[11] == 1; // Check c.a[c.y + 1].
        if(c.a[c.y + 1] == 1) {
            lock_requests[11] = lock_requests[11] - 1; // Release c.a[c.y + 1]
            //@ assert lock_requests[11] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[c.y + 2]
        //@ assert lock_requests[10] == 0; // Verify lock activity.
        lock_requests[11] = lock_requests[11] - 1; // Release c.a[c.y + 1]
        //@ assert lock_requests[11] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y = 0 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 0 && c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8], 10: a[y + 2]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_0_s_0_n_2() {
        if(t_running_0_s_0_n_0()) {
            if(t_running_0_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8], 10: a[y + 2]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9 || _i == 10) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_0_s_0_n_3() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[10] == 1; // Check c.a[c.y + 2].
        if(c.a[c.y + 2] == 1) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.y
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            lock_requests[1] = lock_requests[1] - 1; // Release c.a[0]
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            lock_requests[2] = lock_requests[2] - 1; // Release c.a[1]
            //@ assert lock_requests[2] == 0; // Verify lock activity.
            lock_requests[3] = lock_requests[3] - 1; // Release c.a[2]
            //@ assert lock_requests[3] == 0; // Verify lock activity.
            lock_requests[4] = lock_requests[4] - 1; // Release c.a[3]
            //@ assert lock_requests[4] == 0; // Verify lock activity.
            lock_requests[5] = lock_requests[5] - 1; // Release c.a[4]
            //@ assert lock_requests[5] == 0; // Verify lock activity.
            lock_requests[6] = lock_requests[6] - 1; // Release c.a[5]
            //@ assert lock_requests[6] == 0; // Verify lock activity.
            lock_requests[7] = lock_requests[7] - 1; // Release c.a[6]
            //@ assert lock_requests[7] == 0; // Verify lock activity.
            lock_requests[8] = lock_requests[8] - 1; // Release c.a[7]
            //@ assert lock_requests[8] == 0; // Verify lock activity.
            lock_requests[9] = lock_requests[9] - 1; // Release c.a[8]
            //@ assert lock_requests[9] == 0; // Verify lock activity.
            lock_requests[10] = lock_requests[10] - 1; // Release c.a[c.y + 2]
            //@ assert lock_requests[10] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[c.y + 2]
        //@ assert lock_requests[10] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 0 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_0_s_0_n_4() {
        if(t_running_0_s_0_n_2()) {
            if(t_running_0_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y + 1 && c.y + 1 < 9;
    ensures 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y + 1 && c.y + 1 < 9;
    requires 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y == 0 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | running -> done | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
    private boolean execute_transition_running_0() {
        // SLCO expression | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
        if(!(t_running_0_s_0_n_4())) {
            return false;
        }

        return true;
    }

    // SLCO expression wrapper | y = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_1_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        return c.y == 1;
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_1_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.a[0].
        //@ assert lock_requests[2] == 1; // Check c.a[1].
        //@ assert lock_requests[3] == 1; // Check c.a[2].
        //@ assert lock_requests[4] == 1; // Check c.a[3].
        //@ assert lock_requests[5] == 1; // Check c.a[4].
        //@ assert lock_requests[6] == 1; // Check c.a[5].
        //@ assert lock_requests[7] == 1; // Check c.a[6].
        //@ assert lock_requests[8] == 1; // Check c.a[7].
        //@ assert lock_requests[9] == 1; // Check c.a[8].
        return c.a[c.y - 1] == 2;
    }

    // SLCO expression wrapper | y = 1 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 1 && c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_1_s_0_n_2() {
        if(t_running_1_s_0_n_0()) {
            if(t_running_1_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_1_s_0_n_3() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.a[0].
        //@ assert lock_requests[2] == 1; // Check c.a[1].
        //@ assert lock_requests[3] == 1; // Check c.a[2].
        //@ assert lock_requests[4] == 1; // Check c.a[3].
        //@ assert lock_requests[5] == 1; // Check c.a[4].
        //@ assert lock_requests[6] == 1; // Check c.a[5].
        //@ assert lock_requests[7] == 1; // Check c.a[6].
        //@ assert lock_requests[8] == 1; // Check c.a[7].
        //@ assert lock_requests[9] == 1; // Check c.a[8].
        return c.a[c.y + 1] == 1;
    }

    // SLCO expression wrapper | y = 1 and a[y - 1] = 2 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 1 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_1_s_0_n_4() {
        if(t_running_1_s_0_n_2()) {
            if(t_running_1_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_1_s_0_n_5() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.a[0].
        //@ assert lock_requests[2] == 1; // Check c.a[1].
        //@ assert lock_requests[3] == 1; // Check c.a[2].
        //@ assert lock_requests[4] == 1; // Check c.a[3].
        //@ assert lock_requests[5] == 1; // Check c.a[4].
        //@ assert lock_requests[6] == 1; // Check c.a[5].
        //@ assert lock_requests[7] == 1; // Check c.a[6].
        //@ assert lock_requests[8] == 1; // Check c.a[7].
        //@ assert lock_requests[9] == 1; // Check c.a[8].
        if(c.a[c.y + 2] == 1) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.y
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            lock_requests[1] = lock_requests[1] - 1; // Release c.a[0]
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            lock_requests[2] = lock_requests[2] - 1; // Release c.a[1]
            //@ assert lock_requests[2] == 0; // Verify lock activity.
            lock_requests[3] = lock_requests[3] - 1; // Release c.a[2]
            //@ assert lock_requests[3] == 0; // Verify lock activity.
            lock_requests[4] = lock_requests[4] - 1; // Release c.a[3]
            //@ assert lock_requests[4] == 0; // Verify lock activity.
            lock_requests[5] = lock_requests[5] - 1; // Release c.a[4]
            //@ assert lock_requests[5] == 0; // Verify lock activity.
            lock_requests[6] = lock_requests[6] - 1; // Release c.a[5]
            //@ assert lock_requests[6] == 0; // Verify lock activity.
            lock_requests[7] = lock_requests[7] - 1; // Release c.a[6]
            //@ assert lock_requests[7] == 0; // Verify lock activity.
            lock_requests[8] = lock_requests[8] - 1; // Release c.a[7]
            //@ assert lock_requests[8] == 0; // Verify lock activity.
            lock_requests[9] = lock_requests[9] - 1; // Release c.a[8]
            //@ assert lock_requests[9] == 0; // Verify lock activity.
            return true;
        }
        return false;
    }

    // SLCO expression wrapper | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;
    context 0 <= c.y + 2 && c.y + 2 < 9;
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 1 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_1_s_0_n_6() {
        if(t_running_1_s_0_n_4()) {
            if(t_running_1_s_0_n_5()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 1 && c.y - 1 < 9;
    ensures 0 <= c.y + 2 && c.y + 2 < 9;
    ensures 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 1 && c.y - 1 < 9;
    requires 0 <= c.y + 2 && c.y + 2 < 9;
    requires 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y == 1 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:1) | running -> done | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
    private boolean execute_transition_running_1() {
        // SLCO expression | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
        if(!(t_running_1_s_0_n_6())) {
            return false;
        }

        return true;
    }

    // SLCO expression wrapper | y = 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 7);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_2_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        return c.y == 7;
    }

    // SLCO expression wrapper | a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_2_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.a[0].
        //@ assert lock_requests[2] == 1; // Check c.a[1].
        //@ assert lock_requests[3] == 1; // Check c.a[2].
        //@ assert lock_requests[4] == 1; // Check c.a[3].
        //@ assert lock_requests[5] == 1; // Check c.a[4].
        //@ assert lock_requests[6] == 1; // Check c.a[5].
        //@ assert lock_requests[7] == 1; // Check c.a[6].
        //@ assert lock_requests[8] == 1; // Check c.a[7].
        //@ assert lock_requests[9] == 1; // Check c.a[8].
        return c.a[c.y - 2] == 2;
    }

    // SLCO expression wrapper | y = 7 and a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 7 && c.a[c.y - 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_2_s_0_n_2() {
        if(t_running_2_s_0_n_0()) {
            if(t_running_2_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_2_s_0_n_3() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.a[0].
        //@ assert lock_requests[2] == 1; // Check c.a[1].
        //@ assert lock_requests[3] == 1; // Check c.a[2].
        //@ assert lock_requests[4] == 1; // Check c.a[3].
        //@ assert lock_requests[5] == 1; // Check c.a[4].
        //@ assert lock_requests[6] == 1; // Check c.a[5].
        //@ assert lock_requests[7] == 1; // Check c.a[6].
        //@ assert lock_requests[8] == 1; // Check c.a[7].
        //@ assert lock_requests[9] == 1; // Check c.a[8].
        return c.a[c.y - 1] == 2;
    }

    // SLCO expression wrapper | y = 7 and a[y - 2] = 2 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_2_s_0_n_4() {
        if(t_running_2_s_0_n_2()) {
            if(t_running_2_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_2_s_0_n_5() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.a[0].
        //@ assert lock_requests[2] == 1; // Check c.a[1].
        //@ assert lock_requests[3] == 1; // Check c.a[2].
        //@ assert lock_requests[4] == 1; // Check c.a[3].
        //@ assert lock_requests[5] == 1; // Check c.a[4].
        //@ assert lock_requests[6] == 1; // Check c.a[5].
        //@ assert lock_requests[7] == 1; // Check c.a[6].
        //@ assert lock_requests[8] == 1; // Check c.a[7].
        //@ assert lock_requests[9] == 1; // Check c.a[8].
        if(c.a[c.y + 1] == 1) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.y
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            lock_requests[1] = lock_requests[1] - 1; // Release c.a[0]
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            lock_requests[2] = lock_requests[2] - 1; // Release c.a[1]
            //@ assert lock_requests[2] == 0; // Verify lock activity.
            lock_requests[3] = lock_requests[3] - 1; // Release c.a[2]
            //@ assert lock_requests[3] == 0; // Verify lock activity.
            lock_requests[4] = lock_requests[4] - 1; // Release c.a[3]
            //@ assert lock_requests[4] == 0; // Verify lock activity.
            lock_requests[5] = lock_requests[5] - 1; // Release c.a[4]
            //@ assert lock_requests[5] == 0; // Verify lock activity.
            lock_requests[6] = lock_requests[6] - 1; // Release c.a[5]
            //@ assert lock_requests[6] == 0; // Verify lock activity.
            lock_requests[7] = lock_requests[7] - 1; // Release c.a[6]
            //@ assert lock_requests[7] == 0; // Verify lock activity.
            lock_requests[8] = lock_requests[8] - 1; // Release c.a[7]
            //@ assert lock_requests[8] == 0; // Verify lock activity.
            lock_requests[9] = lock_requests[9] - 1; // Release c.a[8]
            //@ assert lock_requests[9] == 0; // Verify lock activity.
            return true;
        }
        return false;
    }

    // SLCO expression wrapper | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y + 1 && c.y + 1 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_2_s_0_n_6() {
        if(t_running_2_s_0_n_4()) {
            if(t_running_2_s_0_n_5()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 2 && c.y - 2 < 9;
    ensures 0 <= c.y + 1 && c.y + 1 < 9;
    ensures 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 2 && c.y - 2 < 9;
    requires 0 <= c.y + 1 && c.y + 1 < 9;
    requires 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y == 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:2) | running -> done | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
    private boolean execute_transition_running_2() {
        // SLCO expression | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
        if(!(t_running_2_s_0_n_6())) {
            return false;
        }

        return true;
    }

    // SLCO expression wrapper | y = 8.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 8);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_3_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        return c.y == 8;
    }

    // SLCO expression wrapper | a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_3_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.a[0].
        //@ assert lock_requests[2] == 1; // Check c.a[1].
        //@ assert lock_requests[3] == 1; // Check c.a[2].
        //@ assert lock_requests[4] == 1; // Check c.a[3].
        //@ assert lock_requests[5] == 1; // Check c.a[4].
        //@ assert lock_requests[6] == 1; // Check c.a[5].
        //@ assert lock_requests[7] == 1; // Check c.a[6].
        //@ assert lock_requests[8] == 1; // Check c.a[7].
        //@ assert lock_requests[9] == 1; // Check c.a[8].
        return c.a[c.y - 2] == 2;
    }

    // SLCO expression wrapper | y = 8 and a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 8 && c.a[c.y - 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_3_s_0_n_2() {
        if(t_running_3_s_0_n_0()) {
            if(t_running_3_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_3_s_0_n_3() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.a[0].
        //@ assert lock_requests[2] == 1; // Check c.a[1].
        //@ assert lock_requests[3] == 1; // Check c.a[2].
        //@ assert lock_requests[4] == 1; // Check c.a[3].
        //@ assert lock_requests[5] == 1; // Check c.a[4].
        //@ assert lock_requests[6] == 1; // Check c.a[5].
        //@ assert lock_requests[7] == 1; // Check c.a[6].
        //@ assert lock_requests[8] == 1; // Check c.a[7].
        //@ assert lock_requests[9] == 1; // Check c.a[8].
        if(c.a[c.y - 1] == 2) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.y
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            lock_requests[1] = lock_requests[1] - 1; // Release c.a[0]
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            lock_requests[2] = lock_requests[2] - 1; // Release c.a[1]
            //@ assert lock_requests[2] == 0; // Verify lock activity.
            lock_requests[3] = lock_requests[3] - 1; // Release c.a[2]
            //@ assert lock_requests[3] == 0; // Verify lock activity.
            lock_requests[4] = lock_requests[4] - 1; // Release c.a[3]
            //@ assert lock_requests[4] == 0; // Verify lock activity.
            lock_requests[5] = lock_requests[5] - 1; // Release c.a[4]
            //@ assert lock_requests[5] == 0; // Verify lock activity.
            lock_requests[6] = lock_requests[6] - 1; // Release c.a[5]
            //@ assert lock_requests[6] == 0; // Verify lock activity.
            lock_requests[7] = lock_requests[7] - 1; // Release c.a[6]
            //@ assert lock_requests[7] == 0; // Verify lock activity.
            lock_requests[8] = lock_requests[8] - 1; // Release c.a[7]
            //@ assert lock_requests[8] == 0; // Verify lock activity.
            lock_requests[9] = lock_requests[9] - 1; // Release c.a[8]
            //@ assert lock_requests[9] == 0; // Verify lock activity.
            return true;
        }
        return false;
    }

    // SLCO expression wrapper | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 8 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_running_3_s_0_n_4() {
        if(t_running_3_s_0_n_2()) {
            if(t_running_3_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 2 && c.y - 2 < 9;
    ensures 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 2 && c.y - 2 < 9;
    requires 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y == 8 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:3) | running -> done | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
    private boolean execute_transition_running_3() {
        // SLCO expression | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
        if(!(t_running_3_s_0_n_4())) {
            return false;
        }

        return true;
    }

    // SLCO expression wrapper | y > 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_running_4_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        if(c.y > 1) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.a[0]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[1]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[2]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[3]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[4]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[5]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[6]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[7]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[8]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y < 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_running_4_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        if(c.y < 7) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.a[0]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[1]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[2]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[3]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[4]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[5]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[6]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[7]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[8]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y > 1 and y < 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.y < 7);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_running_4_s_0_n_2() {
        if(t_running_4_s_0_n_0()) {
            if(t_running_4_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_running_4_s_0_n_3() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.a[0].
        //@ assert lock_requests[2] == 1; // Check c.a[1].
        //@ assert lock_requests[3] == 1; // Check c.a[2].
        //@ assert lock_requests[4] == 1; // Check c.a[3].
        //@ assert lock_requests[5] == 1; // Check c.a[4].
        //@ assert lock_requests[6] == 1; // Check c.a[5].
        //@ assert lock_requests[7] == 1; // Check c.a[6].
        //@ assert lock_requests[8] == 1; // Check c.a[7].
        //@ assert lock_requests[9] == 1; // Check c.a[8].
        if(c.a[c.y - 2] == 2) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.a[0]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[1]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[2]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[3]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[4]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[5]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[6]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[7]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[8]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y > 1 and y < 7 and a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.y < 7 && c.a[c.y - 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_running_4_s_0_n_4() {
        if(t_running_4_s_0_n_2()) {
            if(t_running_4_s_0_n_3()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_running_4_s_0_n_5() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.a[0].
        //@ assert lock_requests[2] == 1; // Check c.a[1].
        //@ assert lock_requests[3] == 1; // Check c.a[2].
        //@ assert lock_requests[4] == 1; // Check c.a[3].
        //@ assert lock_requests[5] == 1; // Check c.a[4].
        //@ assert lock_requests[6] == 1; // Check c.a[5].
        //@ assert lock_requests[7] == 1; // Check c.a[6].
        //@ assert lock_requests[8] == 1; // Check c.a[7].
        //@ assert lock_requests[9] == 1; // Check c.a[8].
        if(c.a[c.y - 1] == 2) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.a[0]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[1]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[2]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[3]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[4]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[5]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[6]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[7]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[8]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.y < 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_running_4_s_0_n_6() {
        if(t_running_4_s_0_n_4()) {
            if(t_running_4_s_0_n_5()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_running_4_s_0_n_7() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.a[0].
        //@ assert lock_requests[2] == 1; // Check c.a[1].
        //@ assert lock_requests[3] == 1; // Check c.a[2].
        //@ assert lock_requests[4] == 1; // Check c.a[3].
        //@ assert lock_requests[5] == 1; // Check c.a[4].
        //@ assert lock_requests[6] == 1; // Check c.a[5].
        //@ assert lock_requests[7] == 1; // Check c.a[6].
        //@ assert lock_requests[8] == 1; // Check c.a[7].
        //@ assert lock_requests[9] == 1; // Check c.a[8].
        if(c.a[c.y + 1] == 1) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.a[0]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[1]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[2]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[3]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[4]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[5]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[6]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[7]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[8]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y + 1 && c.y + 1 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.y < 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_running_4_s_0_n_8() {
        if(t_running_4_s_0_n_6()) {
            if(t_running_4_s_0_n_7()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_running_4_s_0_n_9() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.a[0].
        //@ assert lock_requests[2] == 1; // Check c.a[1].
        //@ assert lock_requests[3] == 1; // Check c.a[2].
        //@ assert lock_requests[4] == 1; // Check c.a[3].
        //@ assert lock_requests[5] == 1; // Check c.a[4].
        //@ assert lock_requests[6] == 1; // Check c.a[5].
        //@ assert lock_requests[7] == 1; // Check c.a[6].
        //@ assert lock_requests[8] == 1; // Check c.a[7].
        //@ assert lock_requests[9] == 1; // Check c.a[8].
        if(c.a[c.y + 2] == 1) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.y
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            lock_requests[1] = lock_requests[1] - 1; // Release c.a[0]
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            lock_requests[2] = lock_requests[2] - 1; // Release c.a[1]
            //@ assert lock_requests[2] == 0; // Verify lock activity.
            lock_requests[3] = lock_requests[3] - 1; // Release c.a[2]
            //@ assert lock_requests[3] == 0; // Verify lock activity.
            lock_requests[4] = lock_requests[4] - 1; // Release c.a[3]
            //@ assert lock_requests[4] == 0; // Verify lock activity.
            lock_requests[5] = lock_requests[5] - 1; // Release c.a[4]
            //@ assert lock_requests[5] == 0; // Verify lock activity.
            lock_requests[6] = lock_requests[6] - 1; // Release c.a[5]
            //@ assert lock_requests[6] == 0; // Verify lock activity.
            lock_requests[7] = lock_requests[7] - 1; // Release c.a[6]
            //@ assert lock_requests[7] == 0; // Verify lock activity.
            lock_requests[8] = lock_requests[8] - 1; // Release c.a[7]
            //@ assert lock_requests[8] == 0; // Verify lock activity.
            lock_requests[9] = lock_requests[9] - 1; // Release c.a[8]
            //@ assert lock_requests[9] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.a[0]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.a[1]
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[2]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[3]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[4]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[5]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[6]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[7]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[8]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;
    context 0 <= c.y + 2 && c.y + 2 < 9;
    context 0 <= c.y + 1 && c.y + 1 < 9;
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.y < 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_running_4_s_0_n_10() {
        if(t_running_4_s_0_n_8()) {
            if(t_running_4_s_0_n_9()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 1 && c.y - 1 < 9;
    ensures 0 <= c.y + 2 && c.y + 2 < 9;
    ensures 0 <= c.y + 1 && c.y + 1 < 9;
    ensures 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_4() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 1 && c.y - 1 < 9;
    requires 0 <= c.y + 2 && c.y + 2 < 9;
    requires 0 <= c.y + 1 && c.y + 1 < 9;
    requires 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y > 1 && c.y < 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: a[0], 2: a[1], 3: a[2], 4: a[3], 5: a[4], 6: a[5], 7: a[6], 8: a[7], 9: a[8]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2 || _i == 3 || _i == 4 || _i == 5 || _i == 6 || _i == 7 || _i == 8 || _i == 9) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:4) | running -> done | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
    private boolean execute_transition_running_4() {
        // SLCO expression | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
        if(!(t_running_4_s_0_n_10())) {
            return false;
        }

        return true;
    }

    // SLCO expression wrapper | tmin > y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin > c.y);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 1: tmin]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: fmax]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_done_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.y
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.tmin
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.tmin.
        if(c.tmin > c.y) {
            return true;
        }
        lock_requests[2] = lock_requests[2] + 1; // Acquire c.fmax
        //@ assert lock_requests[2] == 1; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | fmax < y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.fmax < c.y);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: fmax]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_done_0_s_0_n_1() {
        lock_requests[2] = lock_requests[2] + 1; // Acquire c.fmax
        //@ assert lock_requests[2] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.fmax.
        if(c.fmax < c.y) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.y
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            lock_requests[1] = lock_requests[1] - 1; // Release c.tmin
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            lock_requests[2] = lock_requests[2] - 1; // Release c.fmax
            //@ assert lock_requests[2] == 0; // Verify lock activity.
            return true;
        }
        return false;
    }

    // SLCO expression wrapper | tmin > y and fmax < y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin > c.y && c.fmax < c.y);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: fmax]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_done_0_s_0_n_2() {
        if(t_done_0_s_0_n_0()) {
            if(t_done_0_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.tmin > c.y && c.fmax < c.y);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: y, 1: tmin, 2: fmax]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | done -> success | tmin > y and fmax < y.
    private boolean execute_transition_done_0() {
        // SLCO expression | tmin > y and fmax < y.
        if(!(t_done_0_s_0_n_2())) {
            return false;
        }

        return true;
    }

    // SLCO expression wrapper | tmin > y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin > c.y);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: fmax]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: y, 2: fmax]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 2) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_done_1_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[1] == 1; // Check c.tmin.
        if(c.tmin > c.y) {
            lock_requests[1] = lock_requests[1] - 1; // Release c.tmin
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.tmin
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.fmax
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | fmax < y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.fmax < c.y);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 2: fmax]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 2) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_done_1_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.y.
        //@ assert lock_requests[2] == 1; // Check c.fmax.
        if(c.fmax < c.y) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.y
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            lock_requests[2] = lock_requests[2] - 1; // Release c.fmax
            //@ assert lock_requests[2] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[2] = lock_requests[2] - 1; // Release c.fmax
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | tmin > y and fmax < y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin > c.y && c.fmax < c.y);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: fmax]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_done_1_s_0_n_2() {
        if(t_done_1_s_0_n_0()) {
            if(t_done_1_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | !(tmin > y and fmax < y).
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!((c.tmin > c.y && c.fmax < c.y)));

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: fmax]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_done_1_s_0_n_3() {
        return !((t_done_1_s_0_n_2()));
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(!((c.tmin > c.y && c.fmax < c.y)));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: y, 1: tmin, 2: fmax]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1 || _i == 2) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:1) | done -> failure | !(tmin > y and fmax < y).
    private boolean execute_transition_done_1() {
        // SLCO expression | !(tmin > y and fmax < y).
        if(!(t_done_1_s_0_n_3())) {
            return false;
        }

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(true);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | success -> reset | true.
    private boolean execute_transition_success_0() {
        // (Superfluous) SLCO expression | true.

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(true);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | failure -> reset | true.
    private boolean execute_transition_failure_0() {
        // (Superfluous) SLCO expression | true.

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_5() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 4 && 4 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_6() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 4 && 4 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 0 && 0 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_7() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 0 && 0 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 1 && 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_8() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 1 && 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 2 && 2 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_9() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 2 && 2 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 3 && 3 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_10() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 3 && 3 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 5 && 5 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_11() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 5 && 5 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 6 && 6 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_12() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 6 && 6 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 7 && 7 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_13() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 7 && 7 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 8 && 8 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_14() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 8 && 8 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(true);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | reset -> running | true | [true; y := 4; tmin := 0; fmax := 8; a[4] := 0; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2].
    private boolean execute_transition_reset_0() {
        // (Superfluous) SLCO expression | true.

        // SLCO composite | [y := 4; tmin := 0; fmax := 8; a[4] := 0; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2] -> [true; y := 4; tmin := 0; fmax := 8; a[4] := 0; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2].
        // (Superfluous) SLCO expression | true.
        // SLCO assignment | y := 4.
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.y
        //@ assert lock_requests[0] == 1; // Verify lock activity.

        //@ assert lock_requests[0] == 1; // Check c.y.
        range_check_assumption_t_0_s_3();
        c.y = 4;
        lock_requests[0] = lock_requests[0] - 1; // Release c.y
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        // SLCO assignment | tmin := 0.
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.tmin
        //@ assert lock_requests[1] == 1; // Verify lock activity.

        //@ assert lock_requests[1] == 1; // Check c.tmin.
        range_check_assumption_t_0_s_4();
        c.tmin = 0;
        lock_requests[1] = lock_requests[1] - 1; // Release c.tmin
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        // SLCO assignment | fmax := 8.
        lock_requests[2] = lock_requests[2] + 1; // Acquire c.fmax
        //@ assert lock_requests[2] == 1; // Verify lock activity.

        //@ assert lock_requests[2] == 1; // Check c.fmax.
        range_check_assumption_t_0_s_5();
        c.fmax = 8;
        lock_requests[2] = lock_requests[2] - 1; // Release c.fmax
        //@ assert lock_requests[2] == 0; // Verify lock activity.
        // SLCO assignment | a[4] := 0.
        lock_requests[3] = lock_requests[3] + 1; // Acquire c.a[2]
        //@ assert lock_requests[3] == 1; // Verify lock activity.
        lock_requests[4] = lock_requests[4] + 1; // Acquire c.a[0]
        //@ assert lock_requests[4] == 1; // Verify lock activity.
        lock_requests[5] = lock_requests[5] + 1; // Acquire c.a[4]
        //@ assert lock_requests[5] == 1; // Verify lock activity.
        lock_requests[6] = lock_requests[6] + 1; // Acquire c.a[3]
        //@ assert lock_requests[6] == 1; // Verify lock activity.
        lock_requests[7] = lock_requests[7] + 1; // Acquire c.a[1]
        //@ assert lock_requests[7] == 1; // Verify lock activity.

        //@ assert lock_requests[5] == 1; // Check c.a[4].
        range_check_assumption_t_0_s_6();
        c.a[4] = 0;
        lock_requests[5] = lock_requests[5] - 1; // Release c.a[4]
        //@ assert lock_requests[5] == 0; // Verify lock activity.
        // SLCO assignment | a[0] := 1.
        //@ assert lock_requests[4] == 1; // Check c.a[0].
        range_check_assumption_t_0_s_7();
        c.a[0] = 1;
        lock_requests[4] = lock_requests[4] - 1; // Release c.a[0]
        //@ assert lock_requests[4] == 0; // Verify lock activity.
        // SLCO assignment | a[1] := 1.
        //@ assert lock_requests[7] == 1; // Check c.a[1].
        range_check_assumption_t_0_s_8();
        c.a[1] = 1;
        lock_requests[7] = lock_requests[7] - 1; // Release c.a[1]
        //@ assert lock_requests[7] == 0; // Verify lock activity.
        // SLCO assignment | a[2] := 1.
        //@ assert lock_requests[3] == 1; // Check c.a[2].
        range_check_assumption_t_0_s_9();
        c.a[2] = 1;
        lock_requests[3] = lock_requests[3] - 1; // Release c.a[2]
        //@ assert lock_requests[3] == 0; // Verify lock activity.
        // SLCO assignment | a[3] := 1.
        //@ assert lock_requests[6] == 1; // Check c.a[3].
        range_check_assumption_t_0_s_10();
        c.a[3] = 1;
        lock_requests[6] = lock_requests[6] - 1; // Release c.a[3]
        //@ assert lock_requests[6] == 0; // Verify lock activity.
        // SLCO assignment | a[5] := 2.
        lock_requests[8] = lock_requests[8] + 1; // Acquire c.a[5]
        //@ assert lock_requests[8] == 1; // Verify lock activity.

        //@ assert lock_requests[8] == 1; // Check c.a[5].
        range_check_assumption_t_0_s_11();
        c.a[5] = 2;
        lock_requests[8] = lock_requests[8] - 1; // Release c.a[5]
        //@ assert lock_requests[8] == 0; // Verify lock activity.
        // SLCO assignment | a[6] := 2.
        lock_requests[9] = lock_requests[9] + 1; // Acquire c.a[6]
        //@ assert lock_requests[9] == 1; // Verify lock activity.

        //@ assert lock_requests[9] == 1; // Check c.a[6].
        range_check_assumption_t_0_s_12();
        c.a[6] = 2;
        lock_requests[9] = lock_requests[9] - 1; // Release c.a[6]
        //@ assert lock_requests[9] == 0; // Verify lock activity.
        // SLCO assignment | a[7] := 2.
        lock_requests[10] = lock_requests[10] + 1; // Acquire c.a[7]
        //@ assert lock_requests[10] == 1; // Verify lock activity.

        //@ assert lock_requests[10] == 1; // Check c.a[7].
        range_check_assumption_t_0_s_13();
        c.a[7] = 2;
        lock_requests[10] = lock_requests[10] - 1; // Release c.a[7]
        //@ assert lock_requests[10] == 0; // Verify lock activity.
        // SLCO assignment | a[8] := 2.
        lock_requests[11] = lock_requests[11] + 1; // Acquire c.a[8]
        //@ assert lock_requests[11] == 1; // Verify lock activity.

        //@ assert lock_requests[11] == 1; // Check c.a[8].
        range_check_assumption_t_0_s_14();
        c.a[8] = 2;
        lock_requests[11] = lock_requests[11] - 1; // Release c.a[8]
        //@ assert lock_requests[11] == 0; // Verify lock activity.

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state running.
    private void exec_running() {
        // [SEQ.START]
        // [DET.START]
        // SLCO transition (p:0, id:0) | running -> done | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_running_0()) {
            return;
        }
        // SLCO transition (p:0, id:1) | running -> done | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_running_1()) {
            return;
        }
        // SLCO transition (p:0, id:2) | running -> done | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_running_2()) {
            return;
        }
        // SLCO transition (p:0, id:3) | running -> done | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_running_3()) {
            return;
        }
        // SLCO transition (p:0, id:4) | running -> done | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_running_4()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

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
        // [DET.START]
        // SLCO transition (p:0, id:0) | done -> success | tmin > y and fmax < y.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_done_0()) {
            return;
        }
        // SLCO transition (p:0, id:1) | done -> failure | !(tmin > y and fmax < y).
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_done_1()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state success.
    private void exec_success() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | success -> reset | true.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_success_0()) {
            return;
        }
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state failure.
    private void exec_failure() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | failure -> reset | true.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_failure_0()) {
            return;
        }
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 12;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state reset.
    private void exec_reset() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | reset -> running | true | [true; y := 4; tmin := 0; fmax := 8; a[4] := 0; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2].
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_reset_0()) {
            return;
        }
        // [SEQ.END]
    }
}

// <<< STATE_MACHINE.END (control)

// << CLASS.END (GlobalClass)

// < MODEL.END (ToadsAndFrogs)