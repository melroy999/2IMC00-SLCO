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
    }

    // SLCO expression wrapper | v > 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_idle_0_s_0_n_0() {
        //@ assume Perm(c.v, 1);
        return c.v > 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_idle_0_s_0() {
        // SLCO expression | v > 0.
        if(!(t_idle_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | t = p.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_mov_0_s_0_n_0() {
        //@ assume Perm(c.p, 1);
        //@ assume Perm(c.t, 1);
        return c.t == c.p;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_mov_0_s_0() {
        // SLCO expression | t = p.
        if(!(t_mov_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | t < p.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_mov_1_s_0_n_0() {
        //@ assume Perm(c.p, 1);
        //@ assume Perm(c.t, 1);
        return c.t < c.p;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_mov_1_s_0() {
        // SLCO expression | t < p.
        if(!(t_mov_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_mov_1_s_1() {
        // SLCO assignment | p := p - 1.
        //@ assume Perm(c.p, 1);
        c.p = c.p - 1;
    }

    // SLCO expression wrapper | t > p.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_mov_2_s_0_n_0() {
        //@ assume Perm(c.p, 1);
        //@ assume Perm(c.t, 1);
        return c.t > c.p;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_mov_2_s_0() {
        // SLCO expression | t > p.
        if(!(t_mov_2_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_mov_2_s_1() {
        // SLCO assignment | p := p + 1.
        //@ assume Perm(c.p, 1);
        c.p = c.p + 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_open_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_open_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_open_0_s_2() {
        // SLCO assignment | req[p] := 0.
        //@ assume Perm(c.p, 1);
        //@ assume 0 <= c.p && c.p < 4;
        //@ assume Perm(c.req[c.p], 1);
        c.req[c.p] = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_open_0_s_3() {
        // SLCO assignment | v := 0.
        //@ assume Perm(c.v, 1);
        c.v = (0) & 0xff;
    }
}

// <<< STATE_MACHINE.END (cabin)

// >>> STATE_MACHINE.START (environment)

// VerCors verification instructions for SLCO state machine environment.
class GlobalClass_environmentThread {
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
    GlobalClass_environmentThread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;
    }

    // SLCO expression wrapper | req[0] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_read_0_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.req[0], 1);
        return c.req[0] == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_read_0_s_0() {
        // SLCO expression | req[0] = 0.
        if(!(t_read_0_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_read_0_s_1() {
        // SLCO assignment | req[0] := 1.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.req[0], 1);
        c.req[0] = (1) & 0xff;
    }

    // SLCO expression wrapper | req[1] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_read_1_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.req[1], 1);
        return c.req[1] == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_read_1_s_0() {
        // SLCO expression | req[1] = 0.
        if(!(t_read_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_read_1_s_1() {
        // SLCO assignment | req[1] := 1.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.req[1], 1);
        c.req[1] = (1) & 0xff;
    }

    // SLCO expression wrapper | req[2] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_read_2_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.req[2], 1);
        return c.req[2] == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_read_2_s_0() {
        // SLCO expression | req[2] = 0.
        if(!(t_read_2_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_read_2_s_1() {
        // SLCO assignment | req[2] := 1.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.req[2], 1);
        c.req[2] = (1) & 0xff;
    }

    // SLCO expression wrapper | req[3] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_read_3_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.req[3], 1);
        return c.req[3] == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_read_3_s_0() {
        // SLCO expression | req[3] = 0.
        if(!(t_read_3_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.req, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.req != null && c.req.length == 4;
    @*/
    private boolean t_read_3_s_1() {
        // SLCO assignment | req[3] := 1.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.req[3], 1);
        c.req[3] = (1) & 0xff;
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
    @*/
    private boolean t_wait_0_s_0_n_0() {
        //@ assume Perm(c.v, 1);
        return c.v == 0;
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
    @*/
    private boolean t_wait_0_s_0() {
        // SLCO expression | v = 0.
        if(!(t_wait_0_s_0_n_0())) {
            return false;
        }
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
    @*/
    private boolean t_wait_0_s_1() {
        // SLCO assignment | t := t + (2 * ldir) - 1.
        //@ assume Perm(c.t, 1);
        c.t = c.t + (2 * ldir) - 1;
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
    @*/
    private boolean t_work_0_s_0_n_0() {
        //@ assume Perm(c.t, 1);
        return c.t < 0;
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
    @*/
    private boolean t_work_0_s_0_n_1() {
        //@ assume Perm(c.t, 1);
        return c.t == 4;
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
    @*/
    private boolean t_work_0_s_0_n_2() {
        return t_work_0_s_0_n_0() || t_work_0_s_0_n_1();
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
    @*/
    private boolean t_work_0_s_0() {
        // SLCO expression | t < 0 or t = 4.
        if(!(t_work_0_s_0_n_2())) {
            return false;
        }
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
    @*/
    private boolean t_work_0_s_1() {
        // SLCO assignment | ldir := 1 - ldir.
        ldir = (1 - ldir) & 0xff;
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
    @*/
    private boolean t_work_1_s_0_n_0() {
        //@ assume Perm(c.t, 1);
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
    @*/
    private boolean t_work_1_s_0_n_1() {
        //@ assume Perm(c.t, 1);
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
    @*/
    private boolean t_work_1_s_0_n_2() {
        return t_work_1_s_0_n_0() && t_work_1_s_0_n_1();
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
    @*/
    private boolean t_work_1_s_0_n_3() {
        //@ assume Perm(c.t, 1);
        //@ assume 0 <= c.t && c.t < 4;
        //@ assume Perm(c.req[c.t], 1);
        return c.req[c.t] == 1;
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
    @*/
    private boolean t_work_1_s_0_n_4() {
        return t_work_1_s_0_n_2() && t_work_1_s_0_n_3();
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
    @*/
    private boolean t_work_1_s_0() {
        // SLCO expression | t >= 0 and t < 4 and req[t] = 1.
        if(!(t_work_1_s_0_n_4())) {
            return false;
        }
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
    @*/
    private boolean t_work_2_s_0_n_0() {
        //@ assume Perm(c.t, 1);
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
    @*/
    private boolean t_work_2_s_0_n_1() {
        //@ assume Perm(c.t, 1);
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
    @*/
    private boolean t_work_2_s_0_n_2() {
        return t_work_2_s_0_n_0() && t_work_2_s_0_n_1();
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
    @*/
    private boolean t_work_2_s_0_n_3() {
        //@ assume Perm(c.t, 1);
        //@ assume 0 <= c.t && c.t < 4;
        //@ assume Perm(c.req[0], 1); // Lock ids 1
        //@ assume Perm(c.req[1], 1); // Lock ids 2
        //@ assume Perm(c.req[2], 1); // Lock ids 3
        //@ assume Perm(c.req[3], 1); // Lock ids 4
        return c.req[c.t] == 0;
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
    @*/
    private boolean t_work_2_s_0_n_4() {
        return t_work_2_s_0_n_2() && t_work_2_s_0_n_3();
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
    @*/
    private boolean t_work_2_s_0() {
        // SLCO expression | t >= 0 and t < 4 and req[t] = 0.
        if(!(t_work_2_s_0_n_4())) {
            return false;
        }
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
    @*/
    private boolean t_work_2_s_1() {
        // SLCO assignment | t := t + (2 * ldir) - 1.
        //@ assume Perm(c.t, 1);
        c.t = c.t + (2 * ldir) - 1;
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
    @*/
    private boolean t_done_0_s_0() {
        // (Superfluous) SLCO expression | true.
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
    @*/
    private boolean t_done_0_s_1() {
        // SLCO assignment | [v := 1] -> v := 1.
        //@ assume Perm(c.v, 1);
        c.v = (1) & 0xff;
    }
}

// <<< STATE_MACHINE.END (controller)

// << CLASS.END (GlobalClass)

// < MODEL.END (Elevator)