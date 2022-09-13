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
    @*/
    private boolean t_q_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = (c.y - 1); // Lock a[(y - 1)](a[y])
        //@ assume 0 <= c.y && c.y < 9;
        c.a[c.y] = 1;
        c.y = c.y - 1;
        //@ assert _index == c.y;
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
    @*/
    private boolean t_q_1_s_4_lock_rewrite_check_0() {
        //@ ghost int _index = (c.y - 1); // Lock a[(y - 1)](a[y])
        //@ assume 0 <= c.y && c.y < 9;
        c.a[c.y] = 1;
        c.tmin = c.y;
        c.y = c.y - 1;
        //@ assert _index == c.y;
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
    @*/
    private boolean t_q_2_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = (c.y - 2); // Lock a[(y - 2)](a[y])
        //@ assume 0 <= c.y && c.y < 9;
        c.a[c.y] = 1;
        c.y = c.y - 2;
        //@ assert _index == c.y;
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
    @*/
    private boolean t_q_3_s_4_lock_rewrite_check_0() {
        //@ ghost int _index = (c.y - 2); // Lock a[(y - 2)](a[y])
        //@ assume 0 <= c.y && c.y < 9;
        c.a[c.y] = 1;
        c.tmin = c.y;
        c.y = c.y - 2;
        //@ assert _index == c.y;
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
    @*/
    private boolean t_q_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = (c.y + 1); // Lock a[(y + 1)](a[y])
        //@ assume 0 <= c.y && c.y < 9;
        c.a[c.y] = 2;
        c.y = c.y + 1;
        //@ assert _index == c.y;
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
    @*/
    private boolean t_q_1_s_4_lock_rewrite_check_0() {
        //@ ghost int _index = (c.y + 1); // Lock a[(y + 1)](a[y])
        //@ assume 0 <= c.y && c.y < 9;
        c.a[c.y] = 2;
        c.fmax = c.y;
        c.y = c.y + 1;
        //@ assert _index == c.y;
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
    @*/
    private boolean t_q_2_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = (c.y + 2); // Lock a[(y + 2)](a[y])
        //@ assume 0 <= c.y && c.y < 9;
        c.a[c.y] = 2;
        c.y = c.y + 2;
        //@ assert _index == c.y;
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
    @*/
    private boolean t_q_3_s_4_lock_rewrite_check_0() {
        //@ ghost int _index = (c.y + 2); // Lock a[(y + 2)](a[y])
        //@ assume 0 <= c.y && c.y < 9;
        c.a[c.y] = 2;
        c.fmax = c.y;
        c.y = c.y + 2;
        //@ assert _index == c.y;
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
    @*/
    private boolean t_reset_0_s_6_lock_rewrite_check_0() {
        //@ ghost int _index = 0; // Lock a[0]
        //@ assume 0 <= 4 && 4 < 9;
        c.a[4] = 0;
        //@ assert _index == 0;
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
    @*/
    private boolean t_reset_0_s_7_lock_rewrite_check_1() {
        //@ ghost int _index = 1; // Lock a[1]
        //@ assume 0 <= 4 && 4 < 9;
        c.a[4] = 0;
        //@ assume 0 <= 0 && 0 < 9;
        c.a[0] = 1;
        //@ assert _index == 1;
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
    @*/
    private boolean t_reset_0_s_8_lock_rewrite_check_2() {
        //@ ghost int _index = 2; // Lock a[2]
        //@ assume 0 <= 4 && 4 < 9;
        c.a[4] = 0;
        //@ assume 0 <= 0 && 0 < 9;
        c.a[0] = 1;
        //@ assume 0 <= 1 && 1 < 9;
        c.a[1] = 1;
        //@ assert _index == 2;
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
    @*/
    private boolean t_reset_0_s_9_lock_rewrite_check_3() {
        //@ ghost int _index = 3; // Lock a[3]
        //@ assume 0 <= 4 && 4 < 9;
        c.a[4] = 0;
        //@ assume 0 <= 0 && 0 < 9;
        c.a[0] = 1;
        //@ assume 0 <= 1 && 1 < 9;
        c.a[1] = 1;
        //@ assume 0 <= 2 && 2 < 9;
        c.a[2] = 1;
        //@ assert _index == 3;
    }
}

// <<< STATE_MACHINE.END (control)

// << CLASS.END (GlobalClass)

// < MODEL.END (ToadsAndFrogs)