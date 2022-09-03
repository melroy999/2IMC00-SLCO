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
    }

    // SLCO expression wrapper | y > 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_0_n_0() {
        //@ assume Perm(c.y, 1);
        return c.y > 0;
    }

    // SLCO expression wrapper | tmin != y - 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.tmin, 1);
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
    @*/
    private boolean t_q_0_s_0_n_2() {
        return t_q_0_s_0_n_0() && t_q_0_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_0_n_3() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
        //@ assume Perm(c.a[c.y - 1], 1);
        return c.a[c.y - 1] == 1;
    }

    // SLCO expression wrapper | y > 0 and tmin != y - 1 and a[y - 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_0_n_4() {
        return t_q_0_s_0_n_2() && t_q_0_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_0() {
        // SLCO expression | y > 0 and tmin != y - 1 and a[y - 1] = 1.
        if(!(t_q_0_s_0_n_4())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_1() {
        // SLCO assignment | a[y] := 1.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[c.y], 1);
        c.a[c.y] = 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_2() {
        // SLCO assignment | y := y - 1.
        //@ assume Perm(c.y, 1);
        c.y = c.y - 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_3() {
        // SLCO assignment | a[y] := 0.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[c.y], 1);
        c.a[c.y] = 0;
    }

    // SLCO expression wrapper | y > 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_0_n_0() {
        //@ assume Perm(c.y, 1);
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
    @*/
    private boolean t_q_1_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.tmin, 1);
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
    @*/
    private boolean t_q_1_s_0_n_2() {
        return t_q_1_s_0_n_0() && t_q_1_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_0_n_3() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
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
    @*/
    private boolean t_q_1_s_0_n_4() {
        return t_q_1_s_0_n_2() && t_q_1_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_0() {
        // SLCO expression | y > 0 and tmin = y - 1 and a[y - 1] = 1.
        if(!(t_q_1_s_0_n_4())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_1() {
        // SLCO assignment | a[y] := 1.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        c.a[c.y] = 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_2() {
        // SLCO assignment | tmin := y.
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.tmin, 1);
        c.tmin = c.y;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_3() {
        // SLCO assignment | y := y - 1.
        //@ assume Perm(c.y, 1);
        c.y = c.y - 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_4() {
        // SLCO assignment | a[y] := 0.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        c.a[c.y] = 0;
    }

    // SLCO expression wrapper | y > 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_0_n_0() {
        //@ assume Perm(c.y, 1);
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
    @*/
    private boolean t_q_2_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.tmin, 1);
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
    @*/
    private boolean t_q_2_s_0_n_2() {
        return t_q_2_s_0_n_0() && t_q_2_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_0_n_3() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
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
    @*/
    private boolean t_q_2_s_0_n_4() {
        return t_q_2_s_0_n_2() && t_q_2_s_0_n_3();
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_0_n_5() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        return c.a[c.y - 1] == 2;
    }

    // SLCO expression wrapper | y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_0_n_6() {
        return t_q_2_s_0_n_4() && t_q_2_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_0() {
        // SLCO expression | y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
        if(!(t_q_2_s_0_n_6())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_1() {
        // SLCO assignment | a[y] := 1.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        c.a[c.y] = 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_2() {
        // SLCO assignment | y := y - 2.
        //@ assume Perm(c.y, 1);
        c.y = c.y - 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_3() {
        // SLCO assignment | a[y] := 0.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        c.a[c.y] = 0;
    }

    // SLCO expression wrapper | y > 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_0() {
        //@ assume Perm(c.y, 1);
        return c.y > 1;
    }

    // SLCO expression wrapper | tmin = y - 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.tmin, 1);
        return c.tmin == c.y - 2;
    }

    // SLCO expression wrapper | y > 1 and tmin = y - 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_2() {
        return t_q_3_s_0_n_0() && t_q_3_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_3() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        return c.a[c.y - 2] == 1;
    }

    // SLCO expression wrapper | y > 1 and tmin = y - 2 and a[y - 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_4() {
        return t_q_3_s_0_n_2() && t_q_3_s_0_n_3();
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_5() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        return c.a[c.y - 1] == 2;
    }

    // SLCO expression wrapper | y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_6() {
        return t_q_3_s_0_n_4() && t_q_3_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0() {
        // SLCO expression | y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
        if(!(t_q_3_s_0_n_6())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_1() {
        // SLCO assignment | a[y] := 1.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        c.a[c.y] = 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_2() {
        // SLCO assignment | tmin := y.
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.tmin, 1);
        c.tmin = c.y;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_3() {
        // SLCO assignment | y := y - 2.
        //@ assume Perm(c.y, 1);
        c.y = c.y - 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_4() {
        // SLCO assignment | a[y] := 0.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        c.a[c.y] = 0;
    }
}

// <<< STATE_MACHINE.END (toad)

// >>> STATE_MACHINE.START (frog)

// VerCors verification instructions for SLCO state machine frog.
class GlobalClass_frogThread {
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
    GlobalClass_frogThread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;
    }

    // SLCO expression wrapper | y < 8.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_0_n_0() {
        //@ assume Perm(c.y, 1);
        return c.y < 8;
    }

    // SLCO expression wrapper | fmax != y + 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.fmax, 1);
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
    @*/
    private boolean t_q_0_s_0_n_2() {
        return t_q_0_s_0_n_0() && t_q_0_s_0_n_1();
    }

    // SLCO expression wrapper | a[y + 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_0_n_3() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume Perm(c.a[c.y + 1], 1);
        return c.a[c.y + 1] == 2;
    }

    // SLCO expression wrapper | y < 8 and fmax != y + 1 and a[y + 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_0_n_4() {
        return t_q_0_s_0_n_2() && t_q_0_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_0() {
        // SLCO expression | y < 8 and fmax != y + 1 and a[y + 1] = 2.
        if(!(t_q_0_s_0_n_4())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_1() {
        // SLCO assignment | a[y] := 2.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[c.y], 1);
        c.a[c.y] = 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_2() {
        // SLCO assignment | y := y + 1.
        //@ assume Perm(c.y, 1);
        c.y = c.y + 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_0_s_3() {
        // SLCO assignment | a[y] := 0.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[c.y], 1);
        c.a[c.y] = 0;
    }

    // SLCO expression wrapper | y < 8.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_0_n_0() {
        //@ assume Perm(c.y, 1);
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
    @*/
    private boolean t_q_1_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.fmax, 1);
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
    @*/
    private boolean t_q_1_s_0_n_2() {
        return t_q_1_s_0_n_0() && t_q_1_s_0_n_1();
    }

    // SLCO expression wrapper | a[y + 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_0_n_3() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
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
    @*/
    private boolean t_q_1_s_0_n_4() {
        return t_q_1_s_0_n_2() && t_q_1_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_0() {
        // SLCO expression | y < 8 and fmax = y + 1 and a[y + 1] = 2.
        if(!(t_q_1_s_0_n_4())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_1() {
        // SLCO assignment | a[y] := 2.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        c.a[c.y] = 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_2() {
        // SLCO assignment | fmax := y.
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.fmax, 1);
        c.fmax = c.y;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_3() {
        // SLCO assignment | y := y + 1.
        //@ assume Perm(c.y, 1);
        c.y = c.y + 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_1_s_4() {
        // SLCO assignment | a[y] := 0.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        c.a[c.y] = 0;
    }

    // SLCO expression wrapper | y < 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_0_n_0() {
        //@ assume Perm(c.y, 1);
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
    @*/
    private boolean t_q_2_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.fmax, 1);
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
    @*/
    private boolean t_q_2_s_0_n_2() {
        return t_q_2_s_0_n_0() && t_q_2_s_0_n_1();
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_0_n_3() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
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
    @*/
    private boolean t_q_2_s_0_n_4() {
        return t_q_2_s_0_n_2() && t_q_2_s_0_n_3();
    }

    // SLCO expression wrapper | a[y + 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_0_n_5() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        return c.a[c.y + 2] == 2;
    }

    // SLCO expression wrapper | y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_0_n_6() {
        return t_q_2_s_0_n_4() && t_q_2_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_0() {
        // SLCO expression | y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
        if(!(t_q_2_s_0_n_6())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_1() {
        // SLCO assignment | a[y] := 2.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        c.a[c.y] = 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_2() {
        // SLCO assignment | y := y + 2.
        //@ assume Perm(c.y, 1);
        c.y = c.y + 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_2_s_3() {
        // SLCO assignment | a[y] := 0.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        c.a[c.y] = 0;
    }

    // SLCO expression wrapper | y < 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_0() {
        //@ assume Perm(c.y, 1);
        return c.y < 7;
    }

    // SLCO expression wrapper | fmax = y + 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.fmax, 1);
        return c.fmax == c.y + 2;
    }

    // SLCO expression wrapper | y < 7 and fmax = y + 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_2() {
        return t_q_3_s_0_n_0() && t_q_3_s_0_n_1();
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_3() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        return c.a[c.y + 1] == 1;
    }

    // SLCO expression wrapper | y < 7 and fmax = y + 2 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_4() {
        return t_q_3_s_0_n_2() && t_q_3_s_0_n_3();
    }

    // SLCO expression wrapper | a[y + 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_5() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        return c.a[c.y + 2] == 2;
    }

    // SLCO expression wrapper | y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0_n_6() {
        return t_q_3_s_0_n_4() && t_q_3_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_0() {
        // SLCO expression | y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
        if(!(t_q_3_s_0_n_6())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_1() {
        // SLCO assignment | a[y] := 2.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        c.a[c.y] = 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_2() {
        // SLCO assignment | fmax := y.
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.fmax, 1);
        c.fmax = c.y;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_3() {
        // SLCO assignment | y := y + 2.
        //@ assume Perm(c.y, 1);
        c.y = c.y + 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_q_3_s_4() {
        // SLCO assignment | a[y] := 0.
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y && c.y < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 2
        //@ assume Perm(c.a[1], 1); // Lock ids 3
        //@ assume Perm(c.a[2], 1); // Lock ids 4
        //@ assume Perm(c.a[3], 1); // Lock ids 5
        //@ assume Perm(c.a[4], 1); // Lock ids 6
        //@ assume Perm(c.a[5], 1); // Lock ids 7
        //@ assume Perm(c.a[6], 1); // Lock ids 8
        //@ assume Perm(c.a[7], 1); // Lock ids 9
        //@ assume Perm(c.a[8], 1); // Lock ids 10
        c.a[c.y] = 0;
    }
}

// <<< STATE_MACHINE.END (frog)

// >>> STATE_MACHINE.START (control)

// VerCors verification instructions for SLCO state machine control.
class GlobalClass_controlThread {
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
    GlobalClass_controlThread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;
    }

    // SLCO expression wrapper | y = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_0_s_0_n_0() {
        //@ assume Perm(c.y, 1);
        return c.y == 0;
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_0_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume Perm(c.a[c.y + 1], 1);
        return c.a[c.y + 1] == 1;
    }

    // SLCO expression wrapper | y = 0 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_0_s_0_n_2() {
        return t_running_0_s_0_n_0() && t_running_0_s_0_n_1();
    }

    // SLCO expression wrapper | a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_0_s_0_n_3() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
        //@ assume Perm(c.a[c.y + 2], 1);
        return c.a[c.y + 2] == 1;
    }

    // SLCO expression wrapper | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_0_s_0_n_4() {
        return t_running_0_s_0_n_2() && t_running_0_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_0_s_0() {
        // SLCO expression | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
        if(!(t_running_0_s_0_n_4())) {
            return false;
        }
    }

    // SLCO expression wrapper | y = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_1_s_0_n_0() {
        //@ assume Perm(c.y, 1);
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
    @*/
    private boolean t_running_1_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 1
        //@ assume Perm(c.a[1], 1); // Lock ids 2
        //@ assume Perm(c.a[2], 1); // Lock ids 3
        //@ assume Perm(c.a[3], 1); // Lock ids 4
        //@ assume Perm(c.a[4], 1); // Lock ids 5
        //@ assume Perm(c.a[5], 1); // Lock ids 6
        //@ assume Perm(c.a[6], 1); // Lock ids 7
        //@ assume Perm(c.a[7], 1); // Lock ids 8
        //@ assume Perm(c.a[8], 1); // Lock ids 9
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
    @*/
    private boolean t_running_1_s_0_n_2() {
        return t_running_1_s_0_n_0() && t_running_1_s_0_n_1();
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_1_s_0_n_3() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 1
        //@ assume Perm(c.a[1], 1); // Lock ids 2
        //@ assume Perm(c.a[2], 1); // Lock ids 3
        //@ assume Perm(c.a[3], 1); // Lock ids 4
        //@ assume Perm(c.a[4], 1); // Lock ids 5
        //@ assume Perm(c.a[5], 1); // Lock ids 6
        //@ assume Perm(c.a[6], 1); // Lock ids 7
        //@ assume Perm(c.a[7], 1); // Lock ids 8
        //@ assume Perm(c.a[8], 1); // Lock ids 9
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
    @*/
    private boolean t_running_1_s_0_n_4() {
        return t_running_1_s_0_n_2() && t_running_1_s_0_n_3();
    }

    // SLCO expression wrapper | a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_1_s_0_n_5() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 1
        //@ assume Perm(c.a[1], 1); // Lock ids 2
        //@ assume Perm(c.a[2], 1); // Lock ids 3
        //@ assume Perm(c.a[3], 1); // Lock ids 4
        //@ assume Perm(c.a[4], 1); // Lock ids 5
        //@ assume Perm(c.a[5], 1); // Lock ids 6
        //@ assume Perm(c.a[6], 1); // Lock ids 7
        //@ assume Perm(c.a[7], 1); // Lock ids 8
        //@ assume Perm(c.a[8], 1); // Lock ids 9
        return c.a[c.y + 2] == 1;
    }

    // SLCO expression wrapper | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_1_s_0_n_6() {
        return t_running_1_s_0_n_4() && t_running_1_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_1_s_0() {
        // SLCO expression | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
        if(!(t_running_1_s_0_n_6())) {
            return false;
        }
    }

    // SLCO expression wrapper | y = 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_2_s_0_n_0() {
        //@ assume Perm(c.y, 1);
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
    @*/
    private boolean t_running_2_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 1
        //@ assume Perm(c.a[1], 1); // Lock ids 2
        //@ assume Perm(c.a[2], 1); // Lock ids 3
        //@ assume Perm(c.a[3], 1); // Lock ids 4
        //@ assume Perm(c.a[4], 1); // Lock ids 5
        //@ assume Perm(c.a[5], 1); // Lock ids 6
        //@ assume Perm(c.a[6], 1); // Lock ids 7
        //@ assume Perm(c.a[7], 1); // Lock ids 8
        //@ assume Perm(c.a[8], 1); // Lock ids 9
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
    @*/
    private boolean t_running_2_s_0_n_2() {
        return t_running_2_s_0_n_0() && t_running_2_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_2_s_0_n_3() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 1
        //@ assume Perm(c.a[1], 1); // Lock ids 2
        //@ assume Perm(c.a[2], 1); // Lock ids 3
        //@ assume Perm(c.a[3], 1); // Lock ids 4
        //@ assume Perm(c.a[4], 1); // Lock ids 5
        //@ assume Perm(c.a[5], 1); // Lock ids 6
        //@ assume Perm(c.a[6], 1); // Lock ids 7
        //@ assume Perm(c.a[7], 1); // Lock ids 8
        //@ assume Perm(c.a[8], 1); // Lock ids 9
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
    @*/
    private boolean t_running_2_s_0_n_4() {
        return t_running_2_s_0_n_2() && t_running_2_s_0_n_3();
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_2_s_0_n_5() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 1
        //@ assume Perm(c.a[1], 1); // Lock ids 2
        //@ assume Perm(c.a[2], 1); // Lock ids 3
        //@ assume Perm(c.a[3], 1); // Lock ids 4
        //@ assume Perm(c.a[4], 1); // Lock ids 5
        //@ assume Perm(c.a[5], 1); // Lock ids 6
        //@ assume Perm(c.a[6], 1); // Lock ids 7
        //@ assume Perm(c.a[7], 1); // Lock ids 8
        //@ assume Perm(c.a[8], 1); // Lock ids 9
        return c.a[c.y + 1] == 1;
    }

    // SLCO expression wrapper | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_2_s_0_n_6() {
        return t_running_2_s_0_n_4() && t_running_2_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_2_s_0() {
        // SLCO expression | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
        if(!(t_running_2_s_0_n_6())) {
            return false;
        }
    }

    // SLCO expression wrapper | y = 8.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_3_s_0_n_0() {
        //@ assume Perm(c.y, 1);
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
    @*/
    private boolean t_running_3_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 1
        //@ assume Perm(c.a[1], 1); // Lock ids 2
        //@ assume Perm(c.a[2], 1); // Lock ids 3
        //@ assume Perm(c.a[3], 1); // Lock ids 4
        //@ assume Perm(c.a[4], 1); // Lock ids 5
        //@ assume Perm(c.a[5], 1); // Lock ids 6
        //@ assume Perm(c.a[6], 1); // Lock ids 7
        //@ assume Perm(c.a[7], 1); // Lock ids 8
        //@ assume Perm(c.a[8], 1); // Lock ids 9
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
    @*/
    private boolean t_running_3_s_0_n_2() {
        return t_running_3_s_0_n_0() && t_running_3_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_3_s_0_n_3() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 1
        //@ assume Perm(c.a[1], 1); // Lock ids 2
        //@ assume Perm(c.a[2], 1); // Lock ids 3
        //@ assume Perm(c.a[3], 1); // Lock ids 4
        //@ assume Perm(c.a[4], 1); // Lock ids 5
        //@ assume Perm(c.a[5], 1); // Lock ids 6
        //@ assume Perm(c.a[6], 1); // Lock ids 7
        //@ assume Perm(c.a[7], 1); // Lock ids 8
        //@ assume Perm(c.a[8], 1); // Lock ids 9
        return c.a[c.y - 1] == 2;
    }

    // SLCO expression wrapper | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_3_s_0_n_4() {
        return t_running_3_s_0_n_2() && t_running_3_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_3_s_0() {
        // SLCO expression | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
        if(!(t_running_3_s_0_n_4())) {
            return false;
        }
    }

    // SLCO expression wrapper | y > 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_4_s_0_n_0() {
        //@ assume Perm(c.y, 1);
        return c.y > 1;
    }

    // SLCO expression wrapper | y < 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_4_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        return c.y < 7;
    }

    // SLCO expression wrapper | y > 1 and y < 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_4_s_0_n_2() {
        return t_running_4_s_0_n_0() && t_running_4_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_4_s_0_n_3() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 1
        //@ assume Perm(c.a[1], 1); // Lock ids 2
        //@ assume Perm(c.a[2], 1); // Lock ids 3
        //@ assume Perm(c.a[3], 1); // Lock ids 4
        //@ assume Perm(c.a[4], 1); // Lock ids 5
        //@ assume Perm(c.a[5], 1); // Lock ids 6
        //@ assume Perm(c.a[6], 1); // Lock ids 7
        //@ assume Perm(c.a[7], 1); // Lock ids 8
        //@ assume Perm(c.a[8], 1); // Lock ids 9
        return c.a[c.y - 2] == 2;
    }

    // SLCO expression wrapper | y > 1 and y < 7 and a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_4_s_0_n_4() {
        return t_running_4_s_0_n_2() && t_running_4_s_0_n_3();
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_4_s_0_n_5() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 1
        //@ assume Perm(c.a[1], 1); // Lock ids 2
        //@ assume Perm(c.a[2], 1); // Lock ids 3
        //@ assume Perm(c.a[3], 1); // Lock ids 4
        //@ assume Perm(c.a[4], 1); // Lock ids 5
        //@ assume Perm(c.a[5], 1); // Lock ids 6
        //@ assume Perm(c.a[6], 1); // Lock ids 7
        //@ assume Perm(c.a[7], 1); // Lock ids 8
        //@ assume Perm(c.a[8], 1); // Lock ids 9
        return c.a[c.y - 1] == 2;
    }

    // SLCO expression wrapper | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_4_s_0_n_6() {
        return t_running_4_s_0_n_4() && t_running_4_s_0_n_5();
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_4_s_0_n_7() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 1
        //@ assume Perm(c.a[1], 1); // Lock ids 2
        //@ assume Perm(c.a[2], 1); // Lock ids 3
        //@ assume Perm(c.a[3], 1); // Lock ids 4
        //@ assume Perm(c.a[4], 1); // Lock ids 5
        //@ assume Perm(c.a[5], 1); // Lock ids 6
        //@ assume Perm(c.a[6], 1); // Lock ids 7
        //@ assume Perm(c.a[7], 1); // Lock ids 8
        //@ assume Perm(c.a[8], 1); // Lock ids 9
        return c.a[c.y + 1] == 1;
    }

    // SLCO expression wrapper | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_4_s_0_n_8() {
        return t_running_4_s_0_n_6() && t_running_4_s_0_n_7();
    }

    // SLCO expression wrapper | a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_4_s_0_n_9() {
        //@ assume Perm(c.y, 1);
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
        //@ assume Perm(c.a[0], 1); // Lock ids 1
        //@ assume Perm(c.a[1], 1); // Lock ids 2
        //@ assume Perm(c.a[2], 1); // Lock ids 3
        //@ assume Perm(c.a[3], 1); // Lock ids 4
        //@ assume Perm(c.a[4], 1); // Lock ids 5
        //@ assume Perm(c.a[5], 1); // Lock ids 6
        //@ assume Perm(c.a[6], 1); // Lock ids 7
        //@ assume Perm(c.a[7], 1); // Lock ids 8
        //@ assume Perm(c.a[8], 1); // Lock ids 9
        return c.a[c.y + 2] == 1;
    }

    // SLCO expression wrapper | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_4_s_0_n_10() {
        return t_running_4_s_0_n_8() && t_running_4_s_0_n_9();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_running_4_s_0() {
        // SLCO expression | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
        if(!(t_running_4_s_0_n_10())) {
            return false;
        }
    }

    // SLCO expression wrapper | tmin > y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_done_0_s_0_n_0() {
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.tmin, 1);
        return c.tmin > c.y;
    }

    // SLCO expression wrapper | fmax < y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_done_0_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.fmax, 1);
        return c.fmax < c.y;
    }

    // SLCO expression wrapper | tmin > y and fmax < y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_done_0_s_0_n_2() {
        return t_done_0_s_0_n_0() && t_done_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_done_0_s_0() {
        // SLCO expression | tmin > y and fmax < y.
        if(!(t_done_0_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | tmin > y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_done_1_s_0_n_0() {
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.tmin, 1);
        return c.tmin > c.y;
    }

    // SLCO expression wrapper | fmax < y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_done_1_s_0_n_1() {
        //@ assume Perm(c.y, 1);
        //@ assume Perm(c.fmax, 1);
        return c.fmax < c.y;
    }

    // SLCO expression wrapper | tmin > y and fmax < y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_done_1_s_0_n_2() {
        return t_done_1_s_0_n_0() && t_done_1_s_0_n_1();
    }

    // SLCO expression wrapper | !(tmin > y and fmax < y).
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
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
    @*/
    private boolean t_done_1_s_0() {
        // SLCO expression | !(tmin > y and fmax < y).
        if(!(t_done_1_s_0_n_3())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_success_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_failure_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_2() {
        // SLCO assignment | y := 4.
        //@ assume Perm(c.y, 1);
        c.y = 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_3() {
        // SLCO assignment | tmin := 0.
        //@ assume Perm(c.tmin, 1);
        c.tmin = 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_4() {
        // SLCO assignment | fmax := 8.
        //@ assume Perm(c.fmax, 1);
        c.fmax = 8;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_5() {
        // SLCO assignment | a[4] := 0.
        //@ assume 0 <= 4 && 4 < 9;
        //@ assume Perm(c.a[4], 1);
        c.a[4] = 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_6() {
        // SLCO assignment | a[0] := 1.
        //@ assume 0 <= 0 && 0 < 9;
        //@ assume Perm(c.a[0], 1);
        c.a[0] = 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_7() {
        // SLCO assignment | a[1] := 1.
        //@ assume 0 <= 1 && 1 < 9;
        //@ assume Perm(c.a[1], 1);
        c.a[1] = 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_8() {
        // SLCO assignment | a[2] := 1.
        //@ assume 0 <= 2 && 2 < 9;
        //@ assume Perm(c.a[2], 1);
        c.a[2] = 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_9() {
        // SLCO assignment | a[3] := 1.
        //@ assume 0 <= 3 && 3 < 9;
        //@ assume Perm(c.a[3], 1);
        c.a[3] = 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_10() {
        // SLCO assignment | a[5] := 2.
        //@ assume 0 <= 5 && 5 < 9;
        //@ assume Perm(c.a[5], 1);
        c.a[5] = 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_11() {
        // SLCO assignment | a[6] := 2.
        //@ assume 0 <= 6 && 6 < 9;
        //@ assume Perm(c.a[6], 1);
        c.a[6] = 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_12() {
        // SLCO assignment | a[7] := 2.
        //@ assume 0 <= 7 && 7 < 9;
        //@ assume Perm(c.a[7], 1);
        c.a[7] = 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;
    @*/
    private boolean t_reset_0_s_13() {
        // SLCO assignment | a[8] := 2.
        //@ assume 0 <= 8 && 8 < 9;
        //@ assume Perm(c.a[8], 1);
        c.a[8] = 2;
    }
}

// <<< STATE_MACHINE.END (control)

// << CLASS.END (GlobalClass)

// < MODEL.END (ToadsAndFrogs)