// > MODEL.START (Tokens)

// >> CLASS.START (P)

// VerCors verification instructions for SLCO class P.
class P {
    // Class variables.
    private final boolean[] tokens;
    private volatile int a;
    private volatile int b;
    private volatile int c;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.tokens, 1);
    ensures Perm(this.a, 1);
    ensures Perm(this.b, 1);
    ensures Perm(this.c, 1);

    // Require that the given values are not null.
    requires tokens != null;

    // Ensure that the right values are assigned.
    ensures this.tokens == tokens;
    ensures this.a == a;
    ensures this.b == b;
    ensures this.c == c;
    @*/
    P(boolean[] tokens, int a, int b, int c) {
        // Instantiate the class variables.
        this.tokens = tokens;
        this.a = a;
        this.b = b;
        this.c = c;
    }
}

// >>> STATE_MACHINE.START (A)

// VerCors verification instructions for SLCO state machine A.
class P_AThread {
    // The class the state machine is a part of.
    private final P c;

    // Thread local variables.
    private int x;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    P_AThread(P c) {
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        x = 1;
    }

    

    

    

    

    

    
}

// <<< STATE_MACHINE.END (A)

// >>> STATE_MACHINE.START (B)

// VerCors verification instructions for SLCO state machine B.
class P_BThread {
    // The class the state machine is a part of.
    private final P c;

    // Thread local variables.
    private int x;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    P_BThread(P c) {
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        x = 42;
    }

    

    

    

    

    

    
}

// <<< STATE_MACHINE.END (B)

// >>> STATE_MACHINE.START (C)

// VerCors verification instructions for SLCO state machine C.
class P_CThread {
    // The class the state machine is a part of.
    private final P c;

    // Thread local variables.
    private int x;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    P_CThread(P c) {
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        x = 308;
    }

    

    

    

    

    

    
}

// <<< STATE_MACHINE.END (C)

// << CLASS.END (P)

// < MODEL.END (Tokens)