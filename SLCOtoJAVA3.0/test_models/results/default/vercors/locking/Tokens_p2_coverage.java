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

    // SLCO expression wrapper | tokens[0].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_0_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 3;
        //@ assume Perm(c.tokens[0], 1);
        return c.tokens[0];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_0_s_0() {
        // SLCO expression | tokens[0].
        if(!(t_act_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | false.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_0_s_1_n_2() {
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_0_s_1() {
        // SLCO assignment | tokens[0] := false.
        //@ assume 0 <= 0 && 0 < 3;
        //@ assume Perm(c.tokens[0], 1);
        c.tokens[0] = t_act_0_s_1_n_2();
    }

    // SLCO expression wrapper | !tokens[0].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 3;
        //@ assume Perm(c.tokens[0], 1);
        return !(c.tokens[0]);
    }

    // SLCO expression wrapper | x % 10 != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_0_n_1() {
        return Math.floorMod(x, 10) != 0;
    }

    // SLCO expression wrapper | !tokens[0] and x % 10 != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_0_n_2() {
        return t_act_1_s_0_n_0() && t_act_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_0() {
        // SLCO expression | !tokens[0] and x % 10 != 0.
        if(!(t_act_1_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | !tokens[0].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 3;
        //@ assume Perm(c.tokens[0], 1);
        return !(c.tokens[0]);
    }

    // SLCO expression wrapper | x % 10 = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_0_n_1() {
        return Math.floorMod(x, 10) == 0;
    }

    // SLCO expression wrapper | !tokens[0] and x % 10 = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_0_n_2() {
        return t_act_2_s_0_n_0() && t_act_2_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_0() {
        // SLCO expression | !tokens[0] and x % 10 = 0.
        if(!(t_act_2_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | true.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_1_n_4() {
        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_1() {
        // SLCO assignment | tokens[1] := true.
        //@ assume 0 <= 1 && 1 < 3;
        //@ assume Perm(c.tokens[1], 1);
        c.tokens[1] = t_act_2_s_1_n_4();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_update_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_update_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_update_0_s_2() {
        // SLCO assignment | x := (641 * x + 718) % 1009.
        x = Math.floorMod((641 * x + 718), 1009);
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_update_0_s_3() {
        // SLCO assignment | a := a + 1.
        //@ assume Perm(c.a, 1);
        c.a = c.a + 1;
    }

    // SLCO expression wrapper | tokens[0].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_0_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 3;
        //@ assume Perm(c.tokens[0], 1);
        return c.tokens[0];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_0_s_0() {
        // SLCO expression | tokens[0].
        if(!(t_wait_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | false.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_0_s_1_n_2() {
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_0_s_1() {
        // SLCO assignment | tokens[0] := false.
        //@ assume 0 <= 0 && 0 < 3;
        //@ assume Perm(c.tokens[0], 1);
        c.tokens[0] = t_wait_0_s_1_n_2();
    }

    // SLCO expression wrapper | !tokens[1].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_1_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 3;
        //@ assume Perm(c.tokens[1], 1);
        return !(c.tokens[1]);
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_1_s_0() {
        // SLCO expression | !tokens[1].
        if(!(t_wait_1_s_0_n_0())) {
            return false;
        }
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

    // SLCO expression wrapper | tokens[1].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_0_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 3;
        //@ assume Perm(c.tokens[1], 1);
        return c.tokens[1];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_0_s_0() {
        // SLCO expression | tokens[1].
        if(!(t_act_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | false.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_0_s_1_n_2() {
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_0_s_1() {
        // SLCO assignment | tokens[1] := false.
        //@ assume 0 <= 1 && 1 < 3;
        //@ assume Perm(c.tokens[1], 1);
        c.tokens[1] = t_act_0_s_1_n_2();
    }

    // SLCO expression wrapper | !tokens[1].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 3;
        //@ assume Perm(c.tokens[1], 1);
        return !(c.tokens[1]);
    }

    // SLCO expression wrapper | x % 10 != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_0_n_1() {
        return Math.floorMod(x, 10) != 0;
    }

    // SLCO expression wrapper | !tokens[1] and x % 10 != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_0_n_2() {
        return t_act_1_s_0_n_0() && t_act_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_0() {
        // SLCO expression | !tokens[1] and x % 10 != 0.
        if(!(t_act_1_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | !tokens[1].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 3;
        //@ assume Perm(c.tokens[1], 1);
        return !(c.tokens[1]);
    }

    // SLCO expression wrapper | x % 10 = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_0_n_1() {
        return Math.floorMod(x, 10) == 0;
    }

    // SLCO expression wrapper | !tokens[1] and x % 10 = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_0_n_2() {
        return t_act_2_s_0_n_0() && t_act_2_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_0() {
        // SLCO expression | !tokens[1] and x % 10 = 0.
        if(!(t_act_2_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | true.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_1_n_4() {
        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_1() {
        // SLCO assignment | tokens[2] := true.
        //@ assume 0 <= 2 && 2 < 3;
        //@ assume Perm(c.tokens[2], 1);
        c.tokens[2] = t_act_2_s_1_n_4();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_update_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_update_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_update_0_s_2() {
        // SLCO assignment | x := (193 * x + 953) % 1009.
        x = Math.floorMod((193 * x + 953), 1009);
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_update_0_s_3() {
        // SLCO assignment | b := b + 1.
        //@ assume Perm(c.b, 1);
        c.b = c.b + 1;
    }

    // SLCO expression wrapper | tokens[1].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_0_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 3;
        //@ assume Perm(c.tokens[1], 1);
        return c.tokens[1];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_0_s_0() {
        // SLCO expression | tokens[1].
        if(!(t_wait_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | false.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_0_s_1_n_2() {
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_0_s_1() {
        // SLCO assignment | tokens[1] := false.
        //@ assume 0 <= 1 && 1 < 3;
        //@ assume Perm(c.tokens[1], 1);
        c.tokens[1] = t_wait_0_s_1_n_2();
    }

    // SLCO expression wrapper | !tokens[2].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_1_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 3;
        //@ assume Perm(c.tokens[2], 1);
        return !(c.tokens[2]);
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_1_s_0() {
        // SLCO expression | !tokens[2].
        if(!(t_wait_1_s_0_n_0())) {
            return false;
        }
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

    // SLCO expression wrapper | tokens[2].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_0_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 3;
        //@ assume Perm(c.tokens[2], 1);
        return c.tokens[2];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_0_s_0() {
        // SLCO expression | tokens[2].
        if(!(t_act_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | false.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_0_s_1_n_2() {
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_0_s_1() {
        // SLCO assignment | tokens[2] := false.
        //@ assume 0 <= 2 && 2 < 3;
        //@ assume Perm(c.tokens[2], 1);
        c.tokens[2] = t_act_0_s_1_n_2();
    }

    // SLCO expression wrapper | !tokens[2].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 3;
        //@ assume Perm(c.tokens[2], 1);
        return !(c.tokens[2]);
    }

    // SLCO expression wrapper | x % 10 = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_0_n_1() {
        return Math.floorMod(x, 10) == 0;
    }

    // SLCO expression wrapper | !tokens[2] and x % 10 = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_0_n_2() {
        return t_act_1_s_0_n_0() && t_act_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_0() {
        // SLCO expression | !tokens[2] and x % 10 = 0.
        if(!(t_act_1_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | true.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_1_n_4() {
        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_1_s_1() {
        // SLCO assignment | tokens[0] := true.
        //@ assume 0 <= 0 && 0 < 3;
        //@ assume Perm(c.tokens[0], 1);
        c.tokens[0] = t_act_1_s_1_n_4();
    }

    // SLCO expression wrapper | !tokens[2].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 3;
        //@ assume Perm(c.tokens[2], 1);
        return !(c.tokens[2]);
    }

    // SLCO expression wrapper | x % 10 != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_0_n_1() {
        return Math.floorMod(x, 10) != 0;
    }

    // SLCO expression wrapper | !tokens[2] and x % 10 != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_0_n_2() {
        return t_act_2_s_0_n_0() && t_act_2_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_act_2_s_0() {
        // SLCO expression | !tokens[2] and x % 10 != 0.
        if(!(t_act_2_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_update_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_update_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_update_0_s_2() {
        // SLCO assignment | x := (811 * x + 31) % 1009.
        x = Math.floorMod((811 * x + 31), 1009);
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_update_0_s_3() {
        // SLCO assignment | c := c + 1.
        //@ assume Perm(c.c, 1);
        c.c = c.c + 1;
    }

    // SLCO expression wrapper | tokens[2].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_0_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 3;
        //@ assume Perm(c.tokens[2], 1);
        return c.tokens[2];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_0_s_0() {
        // SLCO expression | tokens[2].
        if(!(t_wait_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | false.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_0_s_1_n_2() {
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_0_s_1() {
        // SLCO assignment | tokens[2] := false.
        //@ assume 0 <= 2 && 2 < 3;
        //@ assume Perm(c.tokens[2], 1);
        c.tokens[2] = t_wait_0_s_1_n_2();
    }

    // SLCO expression wrapper | !tokens[0].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_1_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 3;
        //@ assume Perm(c.tokens[0], 1);
        return !(c.tokens[0]);
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;
    @*/
    private boolean t_wait_1_s_0() {
        // SLCO expression | !tokens[0].
        if(!(t_wait_1_s_0_n_0())) {
            return false;
        }
    }
}

// <<< STATE_MACHINE.END (C)

// << CLASS.END (P)

// < MODEL.END (Tokens)