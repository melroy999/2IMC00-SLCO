// VerCors verification instructions for SLCO model Test.
// > MODEL.START (SLCOModel:Test)

// VerCors imitation of the lock manager associated with SLCO class P.
class LockManagerForSlcoClassP {
    // The class the lock manager is made for.
    final SlcoClassP c;

    /*@
    // The locking array is given as a ghost parameter.
    given int[] locks;

    // Require and ensure read access to the target class. Moreover, ensure that the target class remains unchanged.
    context Perm(c, 1\2);
    ensures c == \old(c);

    // Require and ensure that the state machine has access to the array variables within the target class.
    context Perm(c.x, 1\2);

    // Require and ensure that the arrays are not null and of the appropriate size.
    context locks != null && locks.length == 3;
    context c.x != null && c.x.length == 2;

    // Put restrictions on the input variables.
    requires lock_id >= 0 && lock_id < 3;

    // Require and ensure the permission of writing to the target element in the locks array.
    context Perm(locks[lock_id], 1);

    // Require and ensure that the value in the locks array is always a positive value.
    context locks[lock_id] >= 0;

    // Ensure that the locks counter is properly incremented.
    ensures locks[lock_id] == \old(locks[lock_id] + 1);

    // Require and ensure that the right permission is enforced, based on the value in the lock array.
    context lock_id == 0 ==> (locks[lock_id] > 0 ==> Perm(c.i, 1)) ** (locks[lock_id] == 0 ==> Perm(c.i, 0));
    context lock_id >= 1 && lock_id < 3 ==> (locks[lock_id] > 0 ==> Perm(c.x[lock_id - 1], 1)) ** (locks[lock_id] == 0 ==> Perm(c.x[lock_id - 1], 0));
    @*/
    // Lock method.
    void acquire_lock(int lock_id) {
        /*@
        ghost locks[lock_id]++;

        // Assume that the post-condition holds to simulate the effect of Re-entrant locking.
        assume lock_id == 0 ==> (locks[lock_id] > 0 ==> Perm(c.i, 1)) ** (locks[lock_id] == 0 ==> Perm(c.i, 0));
        assume lock_id >= 1 && lock_id < 3 ==> (locks[lock_id] > 0 ==> Perm(c.x[lock_id - 1], 1)) ** (locks[lock_id] == 0 ==> Perm(c.x[lock_id - 1], 0));
        @*/
    }

    /*@
    // The locking array is given as a ghost parameter.
    given int[] locks;

    // Require and ensure read access to the target class. Moreover, ensure that the target class remains unchanged.
    context Perm(c, 1\2);
    ensures c == \old(c);

    // Require and ensure that the state machine has access to the array variables within the target class.
    context Perm(c.x, 1\2);

    // Require and ensure that the arrays are not null and of the appropriate size.
    context locks != null && locks.length == 3;
    context c.x != null && c.x.length == 2;

    // Put restrictions on the input variables.
    requires lock_id >= 0 && lock_id < 3;

    // Require and ensure the permission of writing to the target element in the locks array.
    context Perm(locks[lock_id], 1);

    // Require that the target lock has been acquired at least once beforehand.
    requires locks[lock_id] > 0;

    // Ensure that the locks counter is properly decremented.
    ensures locks[lock_id] == \old(locks[lock_id] - 1);

    // Require and ensure that the right permission is enforced, based on the value in the lock array.
    context lock_id == 0 ==> (locks[lock_id] > 0 ==> Perm(c.i, 1)) ** (locks[lock_id] == 0 ==> Perm(c.i, 0));
    context lock_id >= 1 && lock_id < 3 ==> (locks[lock_id] > 0 ==> Perm(c.x[lock_id - 1], 1)) ** (locks[lock_id] == 0 ==> Perm(c.x[lock_id - 1], 0));
    @*/
    // Unlock method.
    void release_lock(int lock_id) {
        /*@
        ghost locks[lock_id]--;

        // Assume that the post-condition holds to simulate the effect of Re-entrant locking.
        assume lock_id == 0 ==> (locks[lock_id] > 0 ==> Perm(c.i, 1)) ** (locks[lock_id] == 0 ==> Perm(c.i, 0));
        assume lock_id >= 1 && lock_id < 3 ==> (locks[lock_id] > 0 ==> Perm(c.x[lock_id - 1], 1)) ** (locks[lock_id] == 0 ==> Perm(c.x[lock_id - 1], 0));
        @*/
    }
}

// VerCors verification instructions for SLCO class P.
class SlcoClassP {
    // The lock manager of the class.
    final LockManagerForSlcoClassP lm;

    // The class variables.
    int i; // Lock id 0
    final int[] x; // Lock id 1, length 2
}

// VerCors verification instructions for SLCO state machine SM1.
class SlcoStateMachineSM1InSlcoClassP {
    // The class the state machine is a part of.
    private final SlcoClassP c;

    // The lock manager of the target class.
    private final LockManagerForSlcoClassP lm;

    // A list of lock ids and target locks that can be reused
    private final int[] lock_ids;
    private final int[] target_locks;

    // >> TRANSITION.START (Transition:SMC0.P0)

    // SLCO expression wrapper | i >= 0
    private boolean t_SMC0_0_s_0_n_0() {
        target_locks[0] = 0; // Acquire i
        lm.acquire_lock(target_locks[0]);
        if(c.i >= 0) {
            return true;
        }
        lm.release_lock(target_locks[0]);
        return false;
    }

    // SLCO expression wrapper | i < 2
    private boolean t_SMC0_0_s_0_n_1() {
        if(c.i < 2) {
            return true;
        }
        lm.release_lock(target_locks[0]);
        return false;
    }

    // SLCO expression wrapper | i >= 0 and i < 2
    private boolean t_SMC0_0_s_0_n_2() {
        return t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1();
    }

    // SLCO expression wrapper | x[i] = 0
    private boolean t_SMC0_0_s_0_n_3() {
        target_locks[1] = 1 + c.i; // Acquire x[i]
        lm.acquire_lock(target_locks[1]);
        if(c.x[c.i] == 0) {
            lm.release_lock(target_locks[1]);
            lm.release_lock(target_locks[0]);
            return true;
        }
        lm.release_lock(target_locks[1]);
        lm.release_lock(target_locks[0]);
        return false;
    }

    // SLCO expression wrapper | i >= 0 and i < 2 and x[i] = 0
    private boolean t_SMC0_0_s_0_n_4() {
        return t_SMC0_0_s_0_n_2() && t_SMC0_0_s_0_n_3();
    }

    /*@
    // The locking array and all array class variables are given as ghost parameters.
    given int[] locks;

    // Require and ensure that the target locks list is of the right size with full write access.
    context target_locks != null && lock_ids.length == 2;
    context Perm(target_locks[*], 1);

    // Require and ensure that the state machine has read access to the target class and associating locking mechanism.
    context Perm(c, 1\2);
    context Perm(lm, 1\2);
    context Perm(lm.c, 1\2);

    // Require and ensure that the locking mechanism refers to the correct class.
    context lm.c == c;

    // Require and ensure that the state machine has access to the array variables within the target class.

    // Require and ensure that the arrays are not null and of the appropriate size.
    context locks != null && locks.length == ;

    // Require and ensure the permission of writing to all elements in the locks array.
    context Perm(locks[*], 1);

    // Require and ensure that all values in locks start and end at zero to verify that no locks remain active after execution.
    context (\forall int _i; _i >= 0 && _i < locks.length; locks[_i] >= 0);
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0
    private boolean execute_transition_SMC0_0() {
        // SLCO expression | i >= 0 and i < 2 and x[i] = 0
        if(!(t_SMC0_0_s_0_n_4())) {
            return false;
        }

        // currentState = SM1Thread.States.SMC0;
        return true;
    }

    // << TRANSITION.END (Transition:SMC0.P0)

}

// < MODEL.END (SLCOModel:Test)