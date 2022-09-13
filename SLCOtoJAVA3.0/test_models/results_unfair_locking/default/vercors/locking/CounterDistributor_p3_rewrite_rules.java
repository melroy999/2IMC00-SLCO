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

    

    

    

    

    

    

    

    

    

    
}

// <<< STATE_MACHINE.END (Distributor)

// << CLASS.END (CounterDistributorExact)

// < MODEL.END (CounterDistributor)