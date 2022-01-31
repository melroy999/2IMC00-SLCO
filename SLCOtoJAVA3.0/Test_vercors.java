// VerCors verification instructions for SLCO model Test.
// >>> MODEL.START (SLCOModel:Test)
// VerCors verification instructions for SLCO class P.
class P {
    // VerCors imitation of the lock manager associated with SLCO class P.
    // >> LOCKMANAGER.START (Class:P)

    // Define non-array global variables as class variables such that they become heap variables.
    int i; // Lock id 0
    // int[] x; // Lock id 1, length 2

    // Lock method.
    /*@
    // The locking array and all array class variables are given as ghost parameters.
    given int[] locks;
    given int[] x;

    // Require and ensure that the arrays are not null and of the appropriate size.
    context locks != null && locks.length == 3;
    context x != null && x.length == 2;

    // Put restrictions on the input variables.
    requires lock_id >= 0 && lock_id < 3;

    // Require and ensure the permission of writing to the target element in the locks array.
    context Perm(locks[lock_id], 1);

    // Require and ensure that the value in the locks array is always a positive value.
    context locks[lock_id] >= 0;

    // Ensure that the locks counter is properly incremented.
    ensures locks[lock_id] == \old(locks[lock_id] + 1);

    // Require and ensure that the right permission is enforced, based on the value in the lock array.
    context lock_id == 0 ==> (locks[lock_id] > 0 ==> Perm(i, 1)) ** (locks[lock_id] == 0 ==> Perm(i, 0));
    context lock_id >= 1 && lock_id < 3 ==> (locks[lock_id] > 0 ==> Perm(x[lock_id - 1], 1)) ** (locks[lock_id] == 0 ==> Perm(x[lock_id - 1], 0));
    @*/
    void acquire_lock(int lock_id) {
        /*@
        ghost locks[lock_id]++;

        // Assume that the post-condition holds to simulate the effect of Re-entrant locking.
        assume lock_id == 0 ==> (locks[lock_id] > 0 ==> Perm(i, 1)) ** (locks[lock_id] == 0 ==> Perm(i, 0));
        assume lock_id >= 1 && lock_id < 3 ==> (locks[lock_id] > 0 ==> Perm(x[lock_id - 1], 1)) ** (locks[lock_id] == 0 ==> Perm(x[lock_id - 1], 0));
        @*/
    }

    // Unlock method.
    /*@
    // The locking array and all array class variables are given as ghost parameters.
    given int[] locks;
    given int[] x;

    // Require and ensure that the arrays are not null and of the appropriate size.
    context locks != null && locks.length == 3;
    context x != null && x.length == 2;

    // Put restrictions on the input variables.
    requires lock_id >= 0 && lock_id < 3;

    // Require and ensure the permission of writing to the target element in the locks array.
    context Perm(locks[lock_id], 1);

    // Require that the target lock has been acquired at least once beforehand.
    requires locks[lock_id] > 0;

    // Ensure that the locks counter is properly decremented.
    ensures locks[lock_id] == \old(locks[lock_id] - 1);

    // Require and ensure that the right permission is enforced, based on the value in the lock array.
    context lock_id == 0 ==> (locks[lock_id] > 0 ==> Perm(i, 1)) ** (locks[lock_id] == 0 ==> Perm(i, 0));
    context lock_id >= 1 && lock_id < 3 ==> (locks[lock_id] > 0 ==> Perm(x[lock_id - 1], 1)) ** (locks[lock_id] == 0 ==> Perm(x[lock_id - 1], 0));
    @*/
    void release_lock(int lock_id) {
        /*@
        ghost locks[lock_id]--;

        // Assume that the post-condition holds to simulate the effect of Re-entrant locking.
        assume lock_id == 0 ==> (locks[lock_id] > 0 ==> Perm(i, 1)) ** (locks[lock_id] == 0 ==> Perm(i, 0));
        assume lock_id >= 1 && lock_id < 3 ==> (locks[lock_id] > 0 ==> Perm(x[lock_id - 1], 1)) ** (locks[lock_id] == 0 ==> Perm(x[lock_id - 1], 0));
        @*/
    }

    // << LOCKMANAGER.END (Class:P)

    // VerCors verification instructions for SLCO state machine SM1.
    // >> SM.START (StateMachine:SM1)

    // > TRANSITION.START (Transition:SMC0.PP0)

    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0
    private boolean execute_transition_SMC0_0() {
        // currentState = SM1Thread.States.SMC0;
        return true;
    }

    // < TRANSITION.END (Transition:SMC0.PP0)

    // << SM.END (StateMachine:SM1)
}

// <<< MODEL.END (SLCOModel:Test)