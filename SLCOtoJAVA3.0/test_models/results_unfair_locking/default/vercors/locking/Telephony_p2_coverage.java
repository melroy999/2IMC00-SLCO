// > MODEL.START (Telephony)

// >> CLASS.START (GlobalClass)

// VerCors verification instructions for SLCO class GlobalClass.
class GlobalClass {
    // Class variables.
    private final int[] chan;
    private final int[] partner;
    private final int[] callforwardbusy;
    private final int[] record;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.chan, 1);
    ensures Perm(this.partner, 1);
    ensures Perm(this.callforwardbusy, 1);
    ensures Perm(this.record, 1);

    // Require that the given values are not null.
    requires chan != null;
    requires partner != null;
    requires callforwardbusy != null;
    requires record != null;

    // Ensure that the right values are assigned.
    ensures this.chan == chan;
    ensures this.partner == partner;
    ensures this.callforwardbusy == callforwardbusy;
    ensures this.record == record;
    @*/
    GlobalClass(int[] chan, int[] partner, int[] callforwardbusy, int[] record) {
        // Instantiate the class variables.
        this.chan = chan;
        this.partner = partner;
        this.callforwardbusy = callforwardbusy;
        this.record = record;
    }
}

// >>> STATE_MACHINE.START (User_0)

// VerCors verification instructions for SLCO state machine User_0.
class GlobalClass_User_0Thread {
    // The class the state machine is a part of.
    private final GlobalClass c;

    // Thread local variables.
    private int dev;
    private int mbit;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    GlobalClass_User_0Thread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        dev = (char) 1;
        mbit = (char) 0;
    }

    // SLCO expression wrapper | chan[0] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        return c.chan[0] == 255;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_0() {
        // SLCO expression | chan[0] = 255.
        if(!(t_idle_0_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_1() {
        // SLCO assignment | dev := 0.
        dev = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_2() {
        // SLCO assignment | chan[0] := (0 + 0 * 20).
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        c.chan[0] = ((0 + 0 * 20)) & 0xff;
    }

    // SLCO expression wrapper | chan[0] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_1_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        return c.chan[0] != 255;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_1_s_0() {
        // SLCO expression | chan[0] != 255.
        if(!(t_idle_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_1_s_1() {
        // SLCO assignment | partner[0] := (chan[0] % 20).
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume Perm(c.chan[0], 1);
        c.partner[0] = ((Math.floorMod(c.chan[0], 20))) & 0xff;
    }

    // SLCO expression wrapper | (chan[partner[0]] % 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_0_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[c.partner[0]], 1);
        return (Math.floorMod(c.chan[c.partner[0]], 20)) == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_0_s_0() {
        // SLCO expression | ((chan[partner[0]]) % 20) = 0 -> (chan[partner[0]] % 20) = 0.
        if(!(t_qi_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[0]] % 20) != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_1_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[c.partner[0]], 1);
        return (Math.floorMod(c.chan[c.partner[0]], 20)) != 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_1_s_0() {
        // SLCO expression | (chan[partner[0]] % 20) != 0.
        if(!(t_qi_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_1_s_1() {
        // SLCO assignment | partner[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        c.partner[0] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_2() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_3() {
        // SLCO assignment | chan[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        c.chan[0] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_1_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_1_s_1() {
        // SLCO assignment | [partner[0] := 0] -> partner[0] := 0.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        c.partner[0] = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_2_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_2_s_1() {
        // SLCO assignment | [partner[0] := 1] -> partner[0] := 1.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        c.partner[0] = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_3_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_3_s_1() {
        // SLCO assignment | [partner[0] := 2] -> partner[0] := 2.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        c.partner[0] = (2) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_4_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_4_s_1() {
        // SLCO assignment | [partner[0] := 3] -> partner[0] := 3.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        c.partner[0] = (3) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_5_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_5_s_1() {
        // SLCO assignment | [partner[0] := 4] -> partner[0] := 4.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        c.partner[0] = (4) & 0xff;
    }

    // SLCO expression wrapper | partner[0] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_0_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        return c.partner[0] == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_0_s_0() {
        // SLCO expression | partner[0] = 0.
        if(!(t_calling_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | partner[0] = 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_1_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        return c.partner[0] == 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_1_s_0() {
        // SLCO expression | partner[0] = 4.
        if(!(t_calling_1_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | partner[0] = 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_2_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        return c.partner[0] == 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_2_s_0() {
        // SLCO expression | partner[0] = 4.
        if(!(t_calling_2_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | partner[0] != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        return c.partner[0] != 0;
    }

    // SLCO expression wrapper | partner[0] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_1() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        return c.partner[0] != 4;
    }

    // SLCO expression wrapper | partner[0] != 0 and partner[0] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_2() {
        return t_calling_3_s_0_n_0() && t_calling_3_s_0_n_1();
    }

    // SLCO expression wrapper | chan[partner[0]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_3() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[c.partner[0]], 1);
        return c.chan[c.partner[0]] != 255;
    }

    // SLCO expression wrapper | partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_4() {
        return t_calling_3_s_0_n_2() && t_calling_3_s_0_n_3();
    }

    // SLCO expression wrapper | callforwardbusy[partner[0]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_5() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.callforwardbusy[c.partner[0]], 1);
        return c.callforwardbusy[c.partner[0]] == 255;
    }

    // SLCO expression wrapper | partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255 and callforwardbusy[partner[0]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_6() {
        return t_calling_3_s_0_n_4() && t_calling_3_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0() {
        // SLCO expression | partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255 and callforwardbusy[partner[0]] = 255.
        if(!(t_calling_3_s_0_n_6())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_1() {
        // SLCO assignment | record[partner[0]] := 0.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.record[c.partner[0]], 1);
        c.record[c.partner[0]] = (0) & 0xff;
    }

    // SLCO expression wrapper | partner[0] != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        return c.partner[0] != 0;
    }

    // SLCO expression wrapper | partner[0] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_1() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        return c.partner[0] != 4;
    }

    // SLCO expression wrapper | partner[0] != 0 and partner[0] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_2() {
        return t_calling_4_s_0_n_0() && t_calling_4_s_0_n_1();
    }

    // SLCO expression wrapper | chan[partner[0]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_3() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[0], 1); // Lock ids 9
        //@ assume Perm(c.chan[1], 1); // Lock ids 10
        //@ assume Perm(c.chan[2], 1); // Lock ids 11
        //@ assume Perm(c.chan[3], 1); // Lock ids 12
        return c.chan[c.partner[0]] != 255;
    }

    // SLCO expression wrapper | partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_4() {
        return t_calling_4_s_0_n_2() && t_calling_4_s_0_n_3();
    }

    // SLCO expression wrapper | callforwardbusy[partner[0]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_5() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.callforwardbusy[3], 1); // Lock ids 8
        //@ assume Perm(c.callforwardbusy[0], 1); // Lock ids 5
        //@ assume Perm(c.callforwardbusy[1], 1); // Lock ids 6
        //@ assume Perm(c.callforwardbusy[2], 1); // Lock ids 7
        return c.callforwardbusy[c.partner[0]] != 255;
    }

    // SLCO expression wrapper | partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255 and callforwardbusy[partner[0]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_6() {
        return t_calling_4_s_0_n_4() && t_calling_4_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0() {
        // SLCO expression | partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255 and callforwardbusy[partner[0]] != 255.
        if(!(t_calling_4_s_0_n_6())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_1() {
        // SLCO assignment | record[partner[0]] := 0.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.record[0], 1); // Lock ids 1
        //@ assume Perm(c.record[1], 1); // Lock ids 2
        //@ assume Perm(c.record[2], 1); // Lock ids 3
        //@ assume Perm(c.record[3], 1); // Lock ids 4
        c.record[c.partner[0]] = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_2() {
        // SLCO assignment | partner[0] := callforwardbusy[partner[0]].
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.callforwardbusy[3], 1); // Lock ids 8
        //@ assume Perm(c.callforwardbusy[0], 1); // Lock ids 5
        //@ assume Perm(c.callforwardbusy[1], 1); // Lock ids 6
        //@ assume Perm(c.callforwardbusy[2], 1); // Lock ids 7
        c.partner[0] = (c.callforwardbusy[c.partner[0]]) & 0xff;
    }

    // SLCO expression wrapper | partner[0] != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        return c.partner[0] != 0;
    }

    // SLCO expression wrapper | partner[0] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_1() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        return c.partner[0] != 4;
    }

    // SLCO expression wrapper | partner[0] != 0 and partner[0] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_2() {
        return t_calling_5_s_0_n_0() && t_calling_5_s_0_n_1();
    }

    // SLCO expression wrapper | chan[partner[0]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_3() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[0], 1); // Lock ids 9
        //@ assume Perm(c.chan[1], 1); // Lock ids 10
        //@ assume Perm(c.chan[2], 1); // Lock ids 11
        //@ assume Perm(c.chan[3], 1); // Lock ids 12
        return c.chan[c.partner[0]] == 255;
    }

    // SLCO expression wrapper | partner[0] != 0 and partner[0] != 4 and chan[partner[0]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_4() {
        return t_calling_5_s_0_n_2() && t_calling_5_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0() {
        // SLCO expression | partner[0] != 0 and partner[0] != 4 and chan[partner[0]] = 255.
        if(!(t_calling_5_s_0_n_4())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_1() {
        // SLCO assignment | record[partner[0]] := 0.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.record[0], 1); // Lock ids 1
        //@ assume Perm(c.record[1], 1); // Lock ids 2
        //@ assume Perm(c.record[2], 1); // Lock ids 3
        //@ assume Perm(c.record[3], 1); // Lock ids 4
        c.record[c.partner[0]] = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_2() {
        // SLCO assignment | chan[partner[0]] := (0 + 0 * 20).
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[0], 1); // Lock ids 9
        //@ assume Perm(c.chan[1], 1); // Lock ids 10
        //@ assume Perm(c.chan[2], 1); // Lock ids 11
        //@ assume Perm(c.chan[3], 1); // Lock ids 12
        c.chan[c.partner[0]] = ((0 + 0 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_3() {
        // SLCO assignment | chan[0] := (partner[0] + 0 * 20).
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume Perm(c.chan[0], 1);
        c.chan[0] = ((c.partner[0] + 0 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_2() {
        // SLCO assignment | chan[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        c.chan[0] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_3() {
        // SLCO assignment | partner[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        c.partner[0] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_4() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    // SLCO expression wrapper | (chan[0] % 20) != partner[0].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_0_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume Perm(c.chan[0], 1);
        return (Math.floorMod(c.chan[0], 20)) != c.partner[0];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_0_s_0() {
        // SLCO expression | ((chan[0]) % 20) != partner[0] -> (chan[0] % 20) != partner[0].
        if(!(t_oalert_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[0] % 20) = partner[0].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume Perm(c.chan[0], 1);
        return (Math.floorMod(c.chan[0], 20)) == c.partner[0];
    }

    // SLCO expression wrapper | (chan[0] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0_n_1() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        return (c.chan[0] / 20) == 1;
    }

    // SLCO expression wrapper | (chan[0] % 20) = partner[0] and (chan[0] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0_n_2() {
        return t_oalert_1_s_0_n_0() && t_oalert_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0() {
        // SLCO expression | ((chan[0]) % 20) = partner[0] and ((chan[0]) / 20) = 1 -> (chan[0] % 20) = partner[0] and (chan[0] / 20) = 1.
        if(!(t_oalert_1_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[0] % 20) = partner[0].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume Perm(c.chan[0], 1);
        return (Math.floorMod(c.chan[0], 20)) == c.partner[0];
    }

    // SLCO expression wrapper | (chan[0] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0_n_1() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        return (c.chan[0] / 20) == 0;
    }

    // SLCO expression wrapper | (chan[0] % 20) = partner[0] and (chan[0] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0_n_2() {
        return t_oalert_2_s_0_n_0() && t_oalert_2_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0() {
        // SLCO expression | ((chan[0]) % 20) = partner[0] and ((chan[0]) / 20) = 0 -> (chan[0] % 20) = partner[0] and (chan[0] / 20) = 0.
        if(!(t_oalert_2_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_2() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_3() {
        // SLCO assignment | chan[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        c.chan[0] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_4() {
        // SLCO assignment | chan[partner[0]] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[c.partner[0]], 1);
        c.chan[c.partner[0]] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_2() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_3() {
        // SLCO assignment | chan[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        c.chan[0] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_4() {
        // SLCO assignment | partner[0] := ((partner[0] % 20) + 0 * 20).
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        c.partner[0] = (((Math.floorMod(c.partner[0], 20)) + 0 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_2() {
        // SLCO assignment | chan[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        c.chan[0] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_3() {
        // SLCO assignment | partner[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        c.partner[0] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_4() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_2() {
        // SLCO assignment | chan[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        c.chan[0] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_3() {
        // SLCO assignment | partner[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        c.partner[0] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_4() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    // SLCO expression wrapper | record[0] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_1_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.record[0], 1);
        return c.record[0] != 255;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_1_s_0() {
        // SLCO expression | record[0] != 255.
        if(!(t_ringback_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_1_s_1() {
        // SLCO assignment | partner[0] := record[0].
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume Perm(c.record[0], 1);
        c.partner[0] = (c.record[0]) & 0xff;
    }

    // SLCO expression wrapper | dev != 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0_n_0() {
        return dev != 1;
    }

    // SLCO expression wrapper | chan[0] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0_n_1() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        return c.chan[0] == 255;
    }

    // SLCO expression wrapper | dev != 1 or chan[0] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0_n_2() {
        return t_talert_0_s_0_n_0() || t_talert_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0() {
        // SLCO expression | dev != 1 or chan[0] = 255.
        if(!(t_talert_0_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[0]] % 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_1_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[c.partner[0]], 1);
        return (Math.floorMod(c.chan[c.partner[0]], 20)) == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_1_s_0() {
        // SLCO expression | ((chan[partner[0]]) % 20) = 0 -> (chan[partner[0]] % 20) = 0.
        if(!(t_talert_1_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[0]] % 20) != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_2_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[c.partner[0]], 1);
        return (Math.floorMod(c.chan[c.partner[0]], 20)) != 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_2_s_0() {
        // SLCO expression | ((chan[partner[0]]) % 20) != 0 -> (chan[partner[0]] % 20) != 0.
        if(!(t_talert_2_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[0]] % 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[c.partner[0]], 1);
        return (Math.floorMod(c.chan[c.partner[0]], 20)) == 0;
    }

    // SLCO expression wrapper | (chan[partner[0]] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0_n_1() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[c.partner[0]], 1);
        return (c.chan[c.partner[0]] / 20) == 0;
    }

    // SLCO expression wrapper | (chan[partner[0]] % 20) = 0 and (chan[partner[0]] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0_n_2() {
        return t_tpickup_0_s_0_n_0() && t_tpickup_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0() {
        // SLCO expression | (chan[partner[0]] % 20) = 0 and (chan[partner[0]] / 20) = 0.
        if(!(t_tpickup_0_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_1() {
        // SLCO assignment | dev := 0.
        dev = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_2() {
        // SLCO assignment | chan[partner[0]] := (0 + 1 * 20).
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[c.partner[0]], 1);
        c.chan[c.partner[0]] = ((0 + 1 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_3() {
        // SLCO assignment | chan[0] := (partner[0] + 1 * 20).
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume Perm(c.chan[0], 1);
        c.chan[0] = ((c.partner[0] + 1 * 20)) & 0xff;
    }

    // SLCO expression wrapper | chan[partner[0]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[c.partner[0]], 1);
        return c.chan[c.partner[0]] == 255;
    }

    // SLCO expression wrapper | (chan[partner[0]] % 20) != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0_n_1() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        //@ assume Perm(c.chan[c.partner[0]], 1);
        return (Math.floorMod(c.chan[c.partner[0]], 20)) != 0;
    }

    // SLCO expression wrapper | chan[partner[0]] = 255 or (chan[partner[0]] % 20) != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0_n_2() {
        return t_tpickup_1_s_0_n_0() || t_tpickup_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0() {
        // SLCO expression | chan[partner[0]] = 255 or (chan[partner[0]] % 20) != 0.
        if(!(t_tpickup_1_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_1() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_2() {
        // SLCO assignment | partner[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        c.partner[0] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_3() {
        // SLCO assignment | chan[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        c.chan[0] = (255) & 0xff;
    }

    // SLCO expression wrapper | (chan[0] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        return (c.chan[0] / 20) == 1;
    }

    // SLCO expression wrapper | dev = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0_n_1() {
        return dev == 0;
    }

    // SLCO expression wrapper | (chan[0] / 20) = 1 and dev = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0_n_2() {
        return t_tconnected_0_s_0_n_0() && t_tconnected_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0() {
        // SLCO expression | (chan[0] / 20) = 1 and dev = 0.
        if(!(t_tconnected_0_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_1() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    // SLCO expression wrapper | (chan[0] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        return (c.chan[0] / 20) == 1;
    }

    // SLCO expression wrapper | dev = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0_n_1() {
        return dev == 1;
    }

    // SLCO expression wrapper | (chan[0] / 20) = 1 and dev = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0_n_2() {
        return t_tconnected_1_s_0_n_0() && t_tconnected_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0() {
        // SLCO expression | (chan[0] / 20) = 1 and dev = 1.
        if(!(t_tconnected_1_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_1() {
        // SLCO assignment | dev := 0.
        dev = (0) & 0xff;
    }

    // SLCO expression wrapper | (chan[0] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_0_n_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        return (c.chan[0] / 20) == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_0() {
        // SLCO expression | (chan[0] / 20) = 0.
        if(!(t_tconnected_2_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_1() {
        // SLCO assignment | partner[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.partner[0], 1);
        c.partner[0] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_2() {
        // SLCO assignment | chan[0] := 255.
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume Perm(c.chan[0], 1);
        c.chan[0] = (255) & 0xff;
    }
}

// <<< STATE_MACHINE.END (User_0)

// >>> STATE_MACHINE.START (User_1)

// VerCors verification instructions for SLCO state machine User_1.
class GlobalClass_User_1Thread {
    // The class the state machine is a part of.
    private final GlobalClass c;

    // Thread local variables.
    private int dev;
    private int mbit;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    GlobalClass_User_1Thread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        dev = (char) 1;
        mbit = (char) 0;
    }

    // SLCO expression wrapper | chan[1] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        return c.chan[1] == 255;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_0() {
        // SLCO expression | chan[1] = 255.
        if(!(t_idle_0_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_1() {
        // SLCO assignment | dev := 0.
        dev = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_2() {
        // SLCO assignment | chan[1] := (1 + 0 * 20).
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        c.chan[1] = ((1 + 0 * 20)) & 0xff;
    }

    // SLCO expression wrapper | chan[1] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_1_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        return c.chan[1] != 255;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_1_s_0() {
        // SLCO expression | chan[1] != 255.
        if(!(t_idle_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_1_s_1() {
        // SLCO assignment | partner[1] := (chan[1] % 20).
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume Perm(c.chan[1], 1);
        c.partner[1] = ((Math.floorMod(c.chan[1], 20))) & 0xff;
    }

    // SLCO expression wrapper | (chan[partner[1]] % 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_0_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[c.partner[1]], 1);
        return (Math.floorMod(c.chan[c.partner[1]], 20)) == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_0_s_0() {
        // SLCO expression | ((chan[partner[1]]) % 20) = 1 -> (chan[partner[1]] % 20) = 1.
        if(!(t_qi_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[1]] % 20) != 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_1_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[c.partner[1]], 1);
        return (Math.floorMod(c.chan[c.partner[1]], 20)) != 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_1_s_0() {
        // SLCO expression | (chan[partner[1]] % 20) != 1.
        if(!(t_qi_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_1_s_1() {
        // SLCO assignment | partner[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        c.partner[1] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_2() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_3() {
        // SLCO assignment | chan[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        c.chan[1] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_1_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_1_s_1() {
        // SLCO assignment | [partner[1] := 0] -> partner[1] := 0.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        c.partner[1] = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_2_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_2_s_1() {
        // SLCO assignment | [partner[1] := 1] -> partner[1] := 1.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        c.partner[1] = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_3_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_3_s_1() {
        // SLCO assignment | [partner[1] := 2] -> partner[1] := 2.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        c.partner[1] = (2) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_4_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_4_s_1() {
        // SLCO assignment | [partner[1] := 3] -> partner[1] := 3.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        c.partner[1] = (3) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_5_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_5_s_1() {
        // SLCO assignment | [partner[1] := 4] -> partner[1] := 4.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        c.partner[1] = (4) & 0xff;
    }

    // SLCO expression wrapper | partner[1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_0_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        return c.partner[1] == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_0_s_0() {
        // SLCO expression | partner[1] = 1.
        if(!(t_calling_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | partner[1] = 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_1_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        return c.partner[1] == 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_1_s_0() {
        // SLCO expression | partner[1] = 4.
        if(!(t_calling_1_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | partner[1] = 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_2_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        return c.partner[1] == 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_2_s_0() {
        // SLCO expression | partner[1] = 4.
        if(!(t_calling_2_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | partner[1] != 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        return c.partner[1] != 1;
    }

    // SLCO expression wrapper | partner[1] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_1() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        return c.partner[1] != 4;
    }

    // SLCO expression wrapper | partner[1] != 1 and partner[1] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_2() {
        return t_calling_3_s_0_n_0() && t_calling_3_s_0_n_1();
    }

    // SLCO expression wrapper | chan[partner[1]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_3() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[c.partner[1]], 1);
        return c.chan[c.partner[1]] != 255;
    }

    // SLCO expression wrapper | partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_4() {
        return t_calling_3_s_0_n_2() && t_calling_3_s_0_n_3();
    }

    // SLCO expression wrapper | callforwardbusy[partner[1]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_5() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.callforwardbusy[c.partner[1]], 1);
        return c.callforwardbusy[c.partner[1]] == 255;
    }

    // SLCO expression wrapper | partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255 and callforwardbusy[partner[1]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_6() {
        return t_calling_3_s_0_n_4() && t_calling_3_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0() {
        // SLCO expression | partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255 and callforwardbusy[partner[1]] = 255.
        if(!(t_calling_3_s_0_n_6())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_1() {
        // SLCO assignment | record[partner[1]] := 1.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.record[c.partner[1]], 1);
        c.record[c.partner[1]] = (1) & 0xff;
    }

    // SLCO expression wrapper | partner[1] != 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        return c.partner[1] != 1;
    }

    // SLCO expression wrapper | partner[1] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_1() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        return c.partner[1] != 4;
    }

    // SLCO expression wrapper | partner[1] != 1 and partner[1] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_2() {
        return t_calling_4_s_0_n_0() && t_calling_4_s_0_n_1();
    }

    // SLCO expression wrapper | chan[partner[1]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_3() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[1], 1); // Lock ids 9
        //@ assume Perm(c.chan[0], 1); // Lock ids 10
        //@ assume Perm(c.chan[2], 1); // Lock ids 11
        //@ assume Perm(c.chan[3], 1); // Lock ids 12
        return c.chan[c.partner[1]] != 255;
    }

    // SLCO expression wrapper | partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_4() {
        return t_calling_4_s_0_n_2() && t_calling_4_s_0_n_3();
    }

    // SLCO expression wrapper | callforwardbusy[partner[1]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_5() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.callforwardbusy[3], 1); // Lock ids 8
        //@ assume Perm(c.callforwardbusy[0], 1); // Lock ids 5
        //@ assume Perm(c.callforwardbusy[1], 1); // Lock ids 6
        //@ assume Perm(c.callforwardbusy[2], 1); // Lock ids 7
        return c.callforwardbusy[c.partner[1]] != 255;
    }

    // SLCO expression wrapper | partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255 and callforwardbusy[partner[1]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_6() {
        return t_calling_4_s_0_n_4() && t_calling_4_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0() {
        // SLCO expression | partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255 and callforwardbusy[partner[1]] != 255.
        if(!(t_calling_4_s_0_n_6())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_1() {
        // SLCO assignment | record[partner[1]] := 1.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.record[0], 1); // Lock ids 1
        //@ assume Perm(c.record[1], 1); // Lock ids 2
        //@ assume Perm(c.record[2], 1); // Lock ids 3
        //@ assume Perm(c.record[3], 1); // Lock ids 4
        c.record[c.partner[1]] = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_2() {
        // SLCO assignment | partner[1] := callforwardbusy[partner[1]].
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.callforwardbusy[3], 1); // Lock ids 8
        //@ assume Perm(c.callforwardbusy[0], 1); // Lock ids 5
        //@ assume Perm(c.callforwardbusy[1], 1); // Lock ids 6
        //@ assume Perm(c.callforwardbusy[2], 1); // Lock ids 7
        c.partner[1] = (c.callforwardbusy[c.partner[1]]) & 0xff;
    }

    // SLCO expression wrapper | partner[1] != 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        return c.partner[1] != 1;
    }

    // SLCO expression wrapper | partner[1] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_1() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        return c.partner[1] != 4;
    }

    // SLCO expression wrapper | partner[1] != 1 and partner[1] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_2() {
        return t_calling_5_s_0_n_0() && t_calling_5_s_0_n_1();
    }

    // SLCO expression wrapper | chan[partner[1]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_3() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[1], 1); // Lock ids 9
        //@ assume Perm(c.chan[0], 1); // Lock ids 10
        //@ assume Perm(c.chan[2], 1); // Lock ids 11
        //@ assume Perm(c.chan[3], 1); // Lock ids 12
        return c.chan[c.partner[1]] == 255;
    }

    // SLCO expression wrapper | partner[1] != 1 and partner[1] != 4 and chan[partner[1]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_4() {
        return t_calling_5_s_0_n_2() && t_calling_5_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0() {
        // SLCO expression | partner[1] != 1 and partner[1] != 4 and chan[partner[1]] = 255.
        if(!(t_calling_5_s_0_n_4())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_1() {
        // SLCO assignment | record[partner[1]] := 1.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.record[0], 1); // Lock ids 1
        //@ assume Perm(c.record[1], 1); // Lock ids 2
        //@ assume Perm(c.record[2], 1); // Lock ids 3
        //@ assume Perm(c.record[3], 1); // Lock ids 4
        c.record[c.partner[1]] = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_2() {
        // SLCO assignment | chan[partner[1]] := (1 + 0 * 20).
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[1], 1); // Lock ids 9
        //@ assume Perm(c.chan[0], 1); // Lock ids 10
        //@ assume Perm(c.chan[2], 1); // Lock ids 11
        //@ assume Perm(c.chan[3], 1); // Lock ids 12
        c.chan[c.partner[1]] = ((1 + 0 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_3() {
        // SLCO assignment | chan[1] := (partner[1] + 0 * 20).
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume Perm(c.chan[1], 1);
        c.chan[1] = ((c.partner[1] + 0 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_2() {
        // SLCO assignment | chan[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        c.chan[1] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_3() {
        // SLCO assignment | partner[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        c.partner[1] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_4() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    // SLCO expression wrapper | (chan[1] % 20) != partner[1].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_0_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume Perm(c.chan[1], 1);
        return (Math.floorMod(c.chan[1], 20)) != c.partner[1];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_0_s_0() {
        // SLCO expression | ((chan[1]) % 20) != partner[1] -> (chan[1] % 20) != partner[1].
        if(!(t_oalert_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[1] % 20) = partner[1].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume Perm(c.chan[1], 1);
        return (Math.floorMod(c.chan[1], 20)) == c.partner[1];
    }

    // SLCO expression wrapper | (chan[1] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0_n_1() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        return (c.chan[1] / 20) == 1;
    }

    // SLCO expression wrapper | (chan[1] % 20) = partner[1] and (chan[1] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0_n_2() {
        return t_oalert_1_s_0_n_0() && t_oalert_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0() {
        // SLCO expression | ((chan[1]) % 20) = partner[1] and ((chan[1]) / 20) = 1 -> (chan[1] % 20) = partner[1] and (chan[1] / 20) = 1.
        if(!(t_oalert_1_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[1] % 20) = partner[1].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume Perm(c.chan[1], 1);
        return (Math.floorMod(c.chan[1], 20)) == c.partner[1];
    }

    // SLCO expression wrapper | (chan[1] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0_n_1() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        return (c.chan[1] / 20) == 0;
    }

    // SLCO expression wrapper | (chan[1] % 20) = partner[1] and (chan[1] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0_n_2() {
        return t_oalert_2_s_0_n_0() && t_oalert_2_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0() {
        // SLCO expression | ((chan[1]) % 20) = partner[1] and ((chan[1]) / 20) = 0 -> (chan[1] % 20) = partner[1] and (chan[1] / 20) = 0.
        if(!(t_oalert_2_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_2() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_3() {
        // SLCO assignment | chan[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        c.chan[1] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_4() {
        // SLCO assignment | chan[partner[1]] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[c.partner[1]], 1);
        c.chan[c.partner[1]] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_2() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_3() {
        // SLCO assignment | chan[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        c.chan[1] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_4() {
        // SLCO assignment | partner[1] := ((partner[1] % 20) + 0 * 20).
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        c.partner[1] = (((Math.floorMod(c.partner[1], 20)) + 0 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_2() {
        // SLCO assignment | chan[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        c.chan[1] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_3() {
        // SLCO assignment | partner[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        c.partner[1] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_4() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_2() {
        // SLCO assignment | chan[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        c.chan[1] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_3() {
        // SLCO assignment | partner[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        c.partner[1] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_4() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    // SLCO expression wrapper | record[1] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_1_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.record[1], 1);
        return c.record[1] != 255;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_1_s_0() {
        // SLCO expression | record[1] != 255.
        if(!(t_ringback_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_1_s_1() {
        // SLCO assignment | partner[1] := record[1].
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume Perm(c.record[1], 1);
        c.partner[1] = (c.record[1]) & 0xff;
    }

    // SLCO expression wrapper | dev != 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0_n_0() {
        return dev != 1;
    }

    // SLCO expression wrapper | chan[1] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0_n_1() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        return c.chan[1] == 255;
    }

    // SLCO expression wrapper | dev != 1 or chan[1] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0_n_2() {
        return t_talert_0_s_0_n_0() || t_talert_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0() {
        // SLCO expression | dev != 1 or chan[1] = 255.
        if(!(t_talert_0_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[1]] % 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_1_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[c.partner[1]], 1);
        return (Math.floorMod(c.chan[c.partner[1]], 20)) == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_1_s_0() {
        // SLCO expression | ((chan[partner[1]]) % 20) = 1 -> (chan[partner[1]] % 20) = 1.
        if(!(t_talert_1_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[1]] % 20) != 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_2_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[c.partner[1]], 1);
        return (Math.floorMod(c.chan[c.partner[1]], 20)) != 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_2_s_0() {
        // SLCO expression | ((chan[partner[1]]) % 20) != 1 -> (chan[partner[1]] % 20) != 1.
        if(!(t_talert_2_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[1]] % 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[c.partner[1]], 1);
        return (Math.floorMod(c.chan[c.partner[1]], 20)) == 1;
    }

    // SLCO expression wrapper | (chan[partner[1]] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0_n_1() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[c.partner[1]], 1);
        return (c.chan[c.partner[1]] / 20) == 0;
    }

    // SLCO expression wrapper | (chan[partner[1]] % 20) = 1 and (chan[partner[1]] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0_n_2() {
        return t_tpickup_0_s_0_n_0() && t_tpickup_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0() {
        // SLCO expression | (chan[partner[1]] % 20) = 1 and (chan[partner[1]] / 20) = 0.
        if(!(t_tpickup_0_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_1() {
        // SLCO assignment | dev := 0.
        dev = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_2() {
        // SLCO assignment | chan[partner[1]] := (1 + 1 * 20).
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[c.partner[1]], 1);
        c.chan[c.partner[1]] = ((1 + 1 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_3() {
        // SLCO assignment | chan[1] := (partner[1] + 1 * 20).
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume Perm(c.chan[1], 1);
        c.chan[1] = ((c.partner[1] + 1 * 20)) & 0xff;
    }

    // SLCO expression wrapper | chan[partner[1]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[c.partner[1]], 1);
        return c.chan[c.partner[1]] == 255;
    }

    // SLCO expression wrapper | (chan[partner[1]] % 20) != 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0_n_1() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        //@ assume Perm(c.chan[c.partner[1]], 1);
        return (Math.floorMod(c.chan[c.partner[1]], 20)) != 1;
    }

    // SLCO expression wrapper | chan[partner[1]] = 255 or (chan[partner[1]] % 20) != 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0_n_2() {
        return t_tpickup_1_s_0_n_0() || t_tpickup_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0() {
        // SLCO expression | chan[partner[1]] = 255 or (chan[partner[1]] % 20) != 1.
        if(!(t_tpickup_1_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_1() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_2() {
        // SLCO assignment | partner[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        c.partner[1] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_3() {
        // SLCO assignment | chan[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        c.chan[1] = (255) & 0xff;
    }

    // SLCO expression wrapper | (chan[1] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        return (c.chan[1] / 20) == 1;
    }

    // SLCO expression wrapper | dev = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0_n_1() {
        return dev == 0;
    }

    // SLCO expression wrapper | (chan[1] / 20) = 1 and dev = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0_n_2() {
        return t_tconnected_0_s_0_n_0() && t_tconnected_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0() {
        // SLCO expression | (chan[1] / 20) = 1 and dev = 0.
        if(!(t_tconnected_0_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_1() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    // SLCO expression wrapper | (chan[1] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        return (c.chan[1] / 20) == 1;
    }

    // SLCO expression wrapper | dev = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0_n_1() {
        return dev == 1;
    }

    // SLCO expression wrapper | (chan[1] / 20) = 1 and dev = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0_n_2() {
        return t_tconnected_1_s_0_n_0() && t_tconnected_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0() {
        // SLCO expression | (chan[1] / 20) = 1 and dev = 1.
        if(!(t_tconnected_1_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_1() {
        // SLCO assignment | dev := 0.
        dev = (0) & 0xff;
    }

    // SLCO expression wrapper | (chan[1] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_0_n_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        return (c.chan[1] / 20) == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_0() {
        // SLCO expression | (chan[1] / 20) = 0.
        if(!(t_tconnected_2_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_1() {
        // SLCO assignment | partner[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.partner[1], 1);
        c.partner[1] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_2() {
        // SLCO assignment | chan[1] := 255.
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume Perm(c.chan[1], 1);
        c.chan[1] = (255) & 0xff;
    }
}

// <<< STATE_MACHINE.END (User_1)

// >>> STATE_MACHINE.START (User_2)

// VerCors verification instructions for SLCO state machine User_2.
class GlobalClass_User_2Thread {
    // The class the state machine is a part of.
    private final GlobalClass c;

    // Thread local variables.
    private int dev;
    private int mbit;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    GlobalClass_User_2Thread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        dev = (char) 1;
        mbit = (char) 0;
    }

    // SLCO expression wrapper | chan[2] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        return c.chan[2] == 255;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_0() {
        // SLCO expression | chan[2] = 255.
        if(!(t_idle_0_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_1() {
        // SLCO assignment | dev := 0.
        dev = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_2() {
        // SLCO assignment | chan[2] := (2 + 0 * 20).
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        c.chan[2] = ((2 + 0 * 20)) & 0xff;
    }

    // SLCO expression wrapper | chan[2] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_1_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        return c.chan[2] != 255;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_1_s_0() {
        // SLCO expression | chan[2] != 255.
        if(!(t_idle_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_1_s_1() {
        // SLCO assignment | partner[2] := (chan[2] % 20).
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume Perm(c.chan[2], 1);
        c.partner[2] = ((Math.floorMod(c.chan[2], 20))) & 0xff;
    }

    // SLCO expression wrapper | (chan[partner[2]] % 20) = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_0_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[c.partner[2]], 1);
        return (Math.floorMod(c.chan[c.partner[2]], 20)) == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_0_s_0() {
        // SLCO expression | ((chan[partner[2]]) % 20) = 2 -> (chan[partner[2]] % 20) = 2.
        if(!(t_qi_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[2]] % 20) != 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_1_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[c.partner[2]], 1);
        return (Math.floorMod(c.chan[c.partner[2]], 20)) != 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_1_s_0() {
        // SLCO expression | (chan[partner[2]] % 20) != 2.
        if(!(t_qi_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_1_s_1() {
        // SLCO assignment | partner[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        c.partner[2] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_2() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_3() {
        // SLCO assignment | chan[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        c.chan[2] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_1_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_1_s_1() {
        // SLCO assignment | [partner[2] := 0] -> partner[2] := 0.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        c.partner[2] = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_2_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_2_s_1() {
        // SLCO assignment | [partner[2] := 1] -> partner[2] := 1.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        c.partner[2] = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_3_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_3_s_1() {
        // SLCO assignment | [partner[2] := 2] -> partner[2] := 2.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        c.partner[2] = (2) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_4_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_4_s_1() {
        // SLCO assignment | [partner[2] := 3] -> partner[2] := 3.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        c.partner[2] = (3) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_5_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_5_s_1() {
        // SLCO assignment | [partner[2] := 4] -> partner[2] := 4.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        c.partner[2] = (4) & 0xff;
    }

    // SLCO expression wrapper | partner[2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_0_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        return c.partner[2] == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_0_s_0() {
        // SLCO expression | partner[2] = 2.
        if(!(t_calling_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | partner[2] = 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_1_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        return c.partner[2] == 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_1_s_0() {
        // SLCO expression | partner[2] = 4.
        if(!(t_calling_1_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | partner[2] = 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_2_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        return c.partner[2] == 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_2_s_0() {
        // SLCO expression | partner[2] = 4.
        if(!(t_calling_2_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | partner[2] != 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        return c.partner[2] != 2;
    }

    // SLCO expression wrapper | partner[2] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_1() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        return c.partner[2] != 4;
    }

    // SLCO expression wrapper | partner[2] != 2 and partner[2] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_2() {
        return t_calling_3_s_0_n_0() && t_calling_3_s_0_n_1();
    }

    // SLCO expression wrapper | chan[partner[2]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_3() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[c.partner[2]], 1);
        return c.chan[c.partner[2]] != 255;
    }

    // SLCO expression wrapper | partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_4() {
        return t_calling_3_s_0_n_2() && t_calling_3_s_0_n_3();
    }

    // SLCO expression wrapper | callforwardbusy[partner[2]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_5() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.callforwardbusy[c.partner[2]], 1);
        return c.callforwardbusy[c.partner[2]] == 255;
    }

    // SLCO expression wrapper | partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255 and callforwardbusy[partner[2]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_6() {
        return t_calling_3_s_0_n_4() && t_calling_3_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0() {
        // SLCO expression | partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255 and callforwardbusy[partner[2]] = 255.
        if(!(t_calling_3_s_0_n_6())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_1() {
        // SLCO assignment | record[partner[2]] := 2.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.record[c.partner[2]], 1);
        c.record[c.partner[2]] = (2) & 0xff;
    }

    // SLCO expression wrapper | partner[2] != 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        return c.partner[2] != 2;
    }

    // SLCO expression wrapper | partner[2] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_1() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        return c.partner[2] != 4;
    }

    // SLCO expression wrapper | partner[2] != 2 and partner[2] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_2() {
        return t_calling_4_s_0_n_0() && t_calling_4_s_0_n_1();
    }

    // SLCO expression wrapper | chan[partner[2]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_3() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[2], 1); // Lock ids 9
        //@ assume Perm(c.chan[0], 1); // Lock ids 10
        //@ assume Perm(c.chan[1], 1); // Lock ids 11
        //@ assume Perm(c.chan[3], 1); // Lock ids 12
        return c.chan[c.partner[2]] != 255;
    }

    // SLCO expression wrapper | partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_4() {
        return t_calling_4_s_0_n_2() && t_calling_4_s_0_n_3();
    }

    // SLCO expression wrapper | callforwardbusy[partner[2]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_5() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.callforwardbusy[3], 1); // Lock ids 8
        //@ assume Perm(c.callforwardbusy[0], 1); // Lock ids 5
        //@ assume Perm(c.callforwardbusy[1], 1); // Lock ids 6
        //@ assume Perm(c.callforwardbusy[2], 1); // Lock ids 7
        return c.callforwardbusy[c.partner[2]] != 255;
    }

    // SLCO expression wrapper | partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255 and callforwardbusy[partner[2]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_6() {
        return t_calling_4_s_0_n_4() && t_calling_4_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0() {
        // SLCO expression | partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255 and callforwardbusy[partner[2]] != 255.
        if(!(t_calling_4_s_0_n_6())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_1() {
        // SLCO assignment | record[partner[2]] := 2.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.record[0], 1); // Lock ids 1
        //@ assume Perm(c.record[1], 1); // Lock ids 2
        //@ assume Perm(c.record[2], 1); // Lock ids 3
        //@ assume Perm(c.record[3], 1); // Lock ids 4
        c.record[c.partner[2]] = (2) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_2() {
        // SLCO assignment | partner[2] := callforwardbusy[partner[2]].
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.callforwardbusy[3], 1); // Lock ids 8
        //@ assume Perm(c.callforwardbusy[0], 1); // Lock ids 5
        //@ assume Perm(c.callforwardbusy[1], 1); // Lock ids 6
        //@ assume Perm(c.callforwardbusy[2], 1); // Lock ids 7
        c.partner[2] = (c.callforwardbusy[c.partner[2]]) & 0xff;
    }

    // SLCO expression wrapper | partner[2] != 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        return c.partner[2] != 2;
    }

    // SLCO expression wrapper | partner[2] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_1() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        return c.partner[2] != 4;
    }

    // SLCO expression wrapper | partner[2] != 2 and partner[2] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_2() {
        return t_calling_5_s_0_n_0() && t_calling_5_s_0_n_1();
    }

    // SLCO expression wrapper | chan[partner[2]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_3() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[2], 1); // Lock ids 9
        //@ assume Perm(c.chan[0], 1); // Lock ids 10
        //@ assume Perm(c.chan[1], 1); // Lock ids 11
        //@ assume Perm(c.chan[3], 1); // Lock ids 12
        return c.chan[c.partner[2]] == 255;
    }

    // SLCO expression wrapper | partner[2] != 2 and partner[2] != 4 and chan[partner[2]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_4() {
        return t_calling_5_s_0_n_2() && t_calling_5_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0() {
        // SLCO expression | partner[2] != 2 and partner[2] != 4 and chan[partner[2]] = 255.
        if(!(t_calling_5_s_0_n_4())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_1() {
        // SLCO assignment | record[partner[2]] := 2.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.record[0], 1); // Lock ids 1
        //@ assume Perm(c.record[1], 1); // Lock ids 2
        //@ assume Perm(c.record[2], 1); // Lock ids 3
        //@ assume Perm(c.record[3], 1); // Lock ids 4
        c.record[c.partner[2]] = (2) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_2() {
        // SLCO assignment | chan[partner[2]] := (2 + 0 * 20).
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[2], 1); // Lock ids 9
        //@ assume Perm(c.chan[0], 1); // Lock ids 10
        //@ assume Perm(c.chan[1], 1); // Lock ids 11
        //@ assume Perm(c.chan[3], 1); // Lock ids 12
        c.chan[c.partner[2]] = ((2 + 0 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_3() {
        // SLCO assignment | chan[2] := (partner[2] + 0 * 20).
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume Perm(c.chan[2], 1);
        c.chan[2] = ((c.partner[2] + 0 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_2() {
        // SLCO assignment | chan[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        c.chan[2] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_3() {
        // SLCO assignment | partner[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        c.partner[2] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_4() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    // SLCO expression wrapper | (chan[2] % 20) != partner[2].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_0_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume Perm(c.chan[2], 1);
        return (Math.floorMod(c.chan[2], 20)) != c.partner[2];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_0_s_0() {
        // SLCO expression | ((chan[2]) % 20) != partner[2] -> (chan[2] % 20) != partner[2].
        if(!(t_oalert_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[2] % 20) = partner[2].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume Perm(c.chan[2], 1);
        return (Math.floorMod(c.chan[2], 20)) == c.partner[2];
    }

    // SLCO expression wrapper | (chan[2] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0_n_1() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        return (c.chan[2] / 20) == 1;
    }

    // SLCO expression wrapper | (chan[2] % 20) = partner[2] and (chan[2] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0_n_2() {
        return t_oalert_1_s_0_n_0() && t_oalert_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0() {
        // SLCO expression | ((chan[2]) % 20) = partner[2] and ((chan[2]) / 20) = 1 -> (chan[2] % 20) = partner[2] and (chan[2] / 20) = 1.
        if(!(t_oalert_1_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[2] % 20) = partner[2].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume Perm(c.chan[2], 1);
        return (Math.floorMod(c.chan[2], 20)) == c.partner[2];
    }

    // SLCO expression wrapper | (chan[2] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0_n_1() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        return (c.chan[2] / 20) == 0;
    }

    // SLCO expression wrapper | (chan[2] % 20) = partner[2] and (chan[2] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0_n_2() {
        return t_oalert_2_s_0_n_0() && t_oalert_2_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0() {
        // SLCO expression | ((chan[2]) % 20) = partner[2] and ((chan[2]) / 20) = 0 -> (chan[2] % 20) = partner[2] and (chan[2] / 20) = 0.
        if(!(t_oalert_2_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_2() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_3() {
        // SLCO assignment | chan[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        c.chan[2] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_4() {
        // SLCO assignment | chan[partner[2]] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[c.partner[2]], 1);
        c.chan[c.partner[2]] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_2() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_3() {
        // SLCO assignment | chan[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        c.chan[2] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_4() {
        // SLCO assignment | partner[2] := ((partner[2] % 20) + 0 * 20).
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        c.partner[2] = (((Math.floorMod(c.partner[2], 20)) + 0 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_2() {
        // SLCO assignment | chan[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        c.chan[2] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_3() {
        // SLCO assignment | partner[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        c.partner[2] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_4() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_2() {
        // SLCO assignment | chan[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        c.chan[2] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_3() {
        // SLCO assignment | partner[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        c.partner[2] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_4() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    // SLCO expression wrapper | record[2] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_1_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.record[2], 1);
        return c.record[2] != 255;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_1_s_0() {
        // SLCO expression | record[2] != 255.
        if(!(t_ringback_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_1_s_1() {
        // SLCO assignment | partner[2] := record[2].
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume Perm(c.record[2], 1);
        c.partner[2] = (c.record[2]) & 0xff;
    }

    // SLCO expression wrapper | dev != 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0_n_0() {
        return dev != 1;
    }

    // SLCO expression wrapper | chan[2] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0_n_1() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        return c.chan[2] == 255;
    }

    // SLCO expression wrapper | dev != 1 or chan[2] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0_n_2() {
        return t_talert_0_s_0_n_0() || t_talert_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0() {
        // SLCO expression | dev != 1 or chan[2] = 255.
        if(!(t_talert_0_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[2]] % 20) = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_1_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[c.partner[2]], 1);
        return (Math.floorMod(c.chan[c.partner[2]], 20)) == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_1_s_0() {
        // SLCO expression | ((chan[partner[2]]) % 20) = 2 -> (chan[partner[2]] % 20) = 2.
        if(!(t_talert_1_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[2]] % 20) != 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_2_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[c.partner[2]], 1);
        return (Math.floorMod(c.chan[c.partner[2]], 20)) != 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_2_s_0() {
        // SLCO expression | ((chan[partner[2]]) % 20) != 2 -> (chan[partner[2]] % 20) != 2.
        if(!(t_talert_2_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[2]] % 20) = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[c.partner[2]], 1);
        return (Math.floorMod(c.chan[c.partner[2]], 20)) == 2;
    }

    // SLCO expression wrapper | (chan[partner[2]] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0_n_1() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[c.partner[2]], 1);
        return (c.chan[c.partner[2]] / 20) == 0;
    }

    // SLCO expression wrapper | (chan[partner[2]] % 20) = 2 and (chan[partner[2]] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0_n_2() {
        return t_tpickup_0_s_0_n_0() && t_tpickup_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0() {
        // SLCO expression | (chan[partner[2]] % 20) = 2 and (chan[partner[2]] / 20) = 0.
        if(!(t_tpickup_0_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_1() {
        // SLCO assignment | dev := 0.
        dev = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_2() {
        // SLCO assignment | chan[partner[2]] := (2 + 1 * 20).
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[c.partner[2]], 1);
        c.chan[c.partner[2]] = ((2 + 1 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_3() {
        // SLCO assignment | chan[2] := (partner[2] + 1 * 20).
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume Perm(c.chan[2], 1);
        c.chan[2] = ((c.partner[2] + 1 * 20)) & 0xff;
    }

    // SLCO expression wrapper | chan[partner[2]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[c.partner[2]], 1);
        return c.chan[c.partner[2]] == 255;
    }

    // SLCO expression wrapper | (chan[partner[2]] % 20) != 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0_n_1() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        //@ assume Perm(c.chan[c.partner[2]], 1);
        return (Math.floorMod(c.chan[c.partner[2]], 20)) != 2;
    }

    // SLCO expression wrapper | chan[partner[2]] = 255 or (chan[partner[2]] % 20) != 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0_n_2() {
        return t_tpickup_1_s_0_n_0() || t_tpickup_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0() {
        // SLCO expression | chan[partner[2]] = 255 or (chan[partner[2]] % 20) != 2.
        if(!(t_tpickup_1_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_1() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_2() {
        // SLCO assignment | partner[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        c.partner[2] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_3() {
        // SLCO assignment | chan[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        c.chan[2] = (255) & 0xff;
    }

    // SLCO expression wrapper | (chan[2] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        return (c.chan[2] / 20) == 1;
    }

    // SLCO expression wrapper | dev = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0_n_1() {
        return dev == 0;
    }

    // SLCO expression wrapper | (chan[2] / 20) = 1 and dev = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0_n_2() {
        return t_tconnected_0_s_0_n_0() && t_tconnected_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0() {
        // SLCO expression | (chan[2] / 20) = 1 and dev = 0.
        if(!(t_tconnected_0_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_1() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    // SLCO expression wrapper | (chan[2] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        return (c.chan[2] / 20) == 1;
    }

    // SLCO expression wrapper | dev = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0_n_1() {
        return dev == 1;
    }

    // SLCO expression wrapper | (chan[2] / 20) = 1 and dev = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0_n_2() {
        return t_tconnected_1_s_0_n_0() && t_tconnected_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0() {
        // SLCO expression | (chan[2] / 20) = 1 and dev = 1.
        if(!(t_tconnected_1_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_1() {
        // SLCO assignment | dev := 0.
        dev = (0) & 0xff;
    }

    // SLCO expression wrapper | (chan[2] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_0_n_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        return (c.chan[2] / 20) == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_0() {
        // SLCO expression | (chan[2] / 20) = 0.
        if(!(t_tconnected_2_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_1() {
        // SLCO assignment | partner[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.partner[2], 1);
        c.partner[2] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_2() {
        // SLCO assignment | chan[2] := 255.
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume Perm(c.chan[2], 1);
        c.chan[2] = (255) & 0xff;
    }
}

// <<< STATE_MACHINE.END (User_2)

// >>> STATE_MACHINE.START (User_3)

// VerCors verification instructions for SLCO state machine User_3.
class GlobalClass_User_3Thread {
    // The class the state machine is a part of.
    private final GlobalClass c;

    // Thread local variables.
    private int dev;
    private int mbit;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    GlobalClass_User_3Thread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        dev = (char) 1;
        mbit = (char) 0;
    }

    // SLCO expression wrapper | chan[3] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        return c.chan[3] == 255;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_0() {
        // SLCO expression | chan[3] = 255.
        if(!(t_idle_0_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_1() {
        // SLCO assignment | dev := 0.
        dev = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_0_s_2() {
        // SLCO assignment | chan[3] := (3 + 0 * 20).
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        c.chan[3] = ((3 + 0 * 20)) & 0xff;
    }

    // SLCO expression wrapper | chan[3] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_1_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        return c.chan[3] != 255;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_1_s_0() {
        // SLCO expression | chan[3] != 255.
        if(!(t_idle_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_idle_1_s_1() {
        // SLCO assignment | partner[3] := (chan[3] % 20).
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume Perm(c.chan[3], 1);
        c.partner[3] = ((Math.floorMod(c.chan[3], 20))) & 0xff;
    }

    // SLCO expression wrapper | (chan[partner[3]] % 20) = 3.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_0_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[c.partner[3]], 1);
        return (Math.floorMod(c.chan[c.partner[3]], 20)) == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_0_s_0() {
        // SLCO expression | ((chan[partner[3]]) % 20) = 3 -> (chan[partner[3]] % 20) = 3.
        if(!(t_qi_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[3]] % 20) != 3.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_1_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[c.partner[3]], 1);
        return (Math.floorMod(c.chan[c.partner[3]], 20)) != 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_1_s_0() {
        // SLCO expression | (chan[partner[3]] % 20) != 3.
        if(!(t_qi_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_qi_1_s_1() {
        // SLCO assignment | partner[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        c.partner[3] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_2() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_0_s_3() {
        // SLCO assignment | chan[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        c.chan[3] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_1_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_1_s_1() {
        // SLCO assignment | [partner[3] := 0] -> partner[3] := 0.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        c.partner[3] = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_2_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_2_s_1() {
        // SLCO assignment | [partner[3] := 1] -> partner[3] := 1.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        c.partner[3] = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_3_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_3_s_1() {
        // SLCO assignment | [partner[3] := 2] -> partner[3] := 2.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        c.partner[3] = (2) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_4_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_4_s_1() {
        // SLCO assignment | [partner[3] := 3] -> partner[3] := 3.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        c.partner[3] = (3) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_5_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dialing_5_s_1() {
        // SLCO assignment | [partner[3] := 4] -> partner[3] := 4.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        c.partner[3] = (4) & 0xff;
    }

    // SLCO expression wrapper | partner[3] = 3.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_0_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        return c.partner[3] == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_0_s_0() {
        // SLCO expression | partner[3] = 3.
        if(!(t_calling_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | partner[3] = 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_1_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        return c.partner[3] == 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_1_s_0() {
        // SLCO expression | partner[3] = 4.
        if(!(t_calling_1_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | partner[3] = 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_2_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        return c.partner[3] == 4;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_2_s_0() {
        // SLCO expression | partner[3] = 4.
        if(!(t_calling_2_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | partner[3] != 3.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        return c.partner[3] != 3;
    }

    // SLCO expression wrapper | partner[3] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_1() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        return c.partner[3] != 4;
    }

    // SLCO expression wrapper | partner[3] != 3 and partner[3] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_2() {
        return t_calling_3_s_0_n_0() && t_calling_3_s_0_n_1();
    }

    // SLCO expression wrapper | chan[partner[3]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_3() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[c.partner[3]], 1);
        return c.chan[c.partner[3]] != 255;
    }

    // SLCO expression wrapper | partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_4() {
        return t_calling_3_s_0_n_2() && t_calling_3_s_0_n_3();
    }

    // SLCO expression wrapper | callforwardbusy[partner[3]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_5() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.callforwardbusy[c.partner[3]], 1);
        return c.callforwardbusy[c.partner[3]] == 255;
    }

    // SLCO expression wrapper | partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255 and callforwardbusy[partner[3]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0_n_6() {
        return t_calling_3_s_0_n_4() && t_calling_3_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_0() {
        // SLCO expression | partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255 and callforwardbusy[partner[3]] = 255.
        if(!(t_calling_3_s_0_n_6())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_3_s_1() {
        // SLCO assignment | record[partner[3]] := 3.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.record[c.partner[3]], 1);
        c.record[c.partner[3]] = (3) & 0xff;
    }

    // SLCO expression wrapper | partner[3] != 3.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        return c.partner[3] != 3;
    }

    // SLCO expression wrapper | partner[3] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_1() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        return c.partner[3] != 4;
    }

    // SLCO expression wrapper | partner[3] != 3 and partner[3] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_2() {
        return t_calling_4_s_0_n_0() && t_calling_4_s_0_n_1();
    }

    // SLCO expression wrapper | chan[partner[3]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_3() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[3], 1); // Lock ids 9
        //@ assume Perm(c.chan[0], 1); // Lock ids 10
        //@ assume Perm(c.chan[1], 1); // Lock ids 11
        //@ assume Perm(c.chan[2], 1); // Lock ids 12
        return c.chan[c.partner[3]] != 255;
    }

    // SLCO expression wrapper | partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_4() {
        return t_calling_4_s_0_n_2() && t_calling_4_s_0_n_3();
    }

    // SLCO expression wrapper | callforwardbusy[partner[3]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_5() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.callforwardbusy[3], 1); // Lock ids 8
        //@ assume Perm(c.callforwardbusy[0], 1); // Lock ids 5
        //@ assume Perm(c.callforwardbusy[1], 1); // Lock ids 6
        //@ assume Perm(c.callforwardbusy[2], 1); // Lock ids 7
        return c.callforwardbusy[c.partner[3]] != 255;
    }

    // SLCO expression wrapper | partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255 and callforwardbusy[partner[3]] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0_n_6() {
        return t_calling_4_s_0_n_4() && t_calling_4_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_0() {
        // SLCO expression | partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255 and callforwardbusy[partner[3]] != 255.
        if(!(t_calling_4_s_0_n_6())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_1() {
        // SLCO assignment | record[partner[3]] := 3.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.record[0], 1); // Lock ids 1
        //@ assume Perm(c.record[1], 1); // Lock ids 2
        //@ assume Perm(c.record[2], 1); // Lock ids 3
        //@ assume Perm(c.record[3], 1); // Lock ids 4
        c.record[c.partner[3]] = (3) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_4_s_2() {
        // SLCO assignment | partner[3] := callforwardbusy[partner[3]].
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.callforwardbusy[3], 1); // Lock ids 8
        //@ assume Perm(c.callforwardbusy[0], 1); // Lock ids 5
        //@ assume Perm(c.callforwardbusy[1], 1); // Lock ids 6
        //@ assume Perm(c.callforwardbusy[2], 1); // Lock ids 7
        c.partner[3] = (c.callforwardbusy[c.partner[3]]) & 0xff;
    }

    // SLCO expression wrapper | partner[3] != 3.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        return c.partner[3] != 3;
    }

    // SLCO expression wrapper | partner[3] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_1() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        return c.partner[3] != 4;
    }

    // SLCO expression wrapper | partner[3] != 3 and partner[3] != 4.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_2() {
        return t_calling_5_s_0_n_0() && t_calling_5_s_0_n_1();
    }

    // SLCO expression wrapper | chan[partner[3]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_3() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[3], 1); // Lock ids 9
        //@ assume Perm(c.chan[0], 1); // Lock ids 10
        //@ assume Perm(c.chan[1], 1); // Lock ids 11
        //@ assume Perm(c.chan[2], 1); // Lock ids 12
        return c.chan[c.partner[3]] == 255;
    }

    // SLCO expression wrapper | partner[3] != 3 and partner[3] != 4 and chan[partner[3]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0_n_4() {
        return t_calling_5_s_0_n_2() && t_calling_5_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_0() {
        // SLCO expression | partner[3] != 3 and partner[3] != 4 and chan[partner[3]] = 255.
        if(!(t_calling_5_s_0_n_4())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_1() {
        // SLCO assignment | record[partner[3]] := 3.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.record[0], 1); // Lock ids 1
        //@ assume Perm(c.record[1], 1); // Lock ids 2
        //@ assume Perm(c.record[2], 1); // Lock ids 3
        //@ assume Perm(c.record[3], 1); // Lock ids 4
        c.record[c.partner[3]] = (3) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_2() {
        // SLCO assignment | chan[partner[3]] := (3 + 0 * 20).
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[3], 1); // Lock ids 9
        //@ assume Perm(c.chan[0], 1); // Lock ids 10
        //@ assume Perm(c.chan[1], 1); // Lock ids 11
        //@ assume Perm(c.chan[2], 1); // Lock ids 12
        c.chan[c.partner[3]] = ((3 + 0 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_calling_5_s_3() {
        // SLCO assignment | chan[3] := (partner[3] + 0 * 20).
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume Perm(c.chan[3], 1);
        c.chan[3] = ((c.partner[3] + 0 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_2() {
        // SLCO assignment | chan[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        c.chan[3] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_3() {
        // SLCO assignment | partner[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        c.partner[3] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_busy_0_s_4() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    // SLCO expression wrapper | (chan[3] % 20) != partner[3].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_0_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume Perm(c.chan[3], 1);
        return (Math.floorMod(c.chan[3], 20)) != c.partner[3];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_0_s_0() {
        // SLCO expression | ((chan[3]) % 20) != partner[3] -> (chan[3] % 20) != partner[3].
        if(!(t_oalert_0_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[3] % 20) = partner[3].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume Perm(c.chan[3], 1);
        return (Math.floorMod(c.chan[3], 20)) == c.partner[3];
    }

    // SLCO expression wrapper | (chan[3] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0_n_1() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        return (c.chan[3] / 20) == 1;
    }

    // SLCO expression wrapper | (chan[3] % 20) = partner[3] and (chan[3] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0_n_2() {
        return t_oalert_1_s_0_n_0() && t_oalert_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_1_s_0() {
        // SLCO expression | ((chan[3]) % 20) = partner[3] and ((chan[3]) / 20) = 1 -> (chan[3] % 20) = partner[3] and (chan[3] / 20) = 1.
        if(!(t_oalert_1_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[3] % 20) = partner[3].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume Perm(c.chan[3], 1);
        return (Math.floorMod(c.chan[3], 20)) == c.partner[3];
    }

    // SLCO expression wrapper | (chan[3] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0_n_1() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        return (c.chan[3] / 20) == 0;
    }

    // SLCO expression wrapper | (chan[3] % 20) = partner[3] and (chan[3] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0_n_2() {
        return t_oalert_2_s_0_n_0() && t_oalert_2_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oalert_2_s_0() {
        // SLCO expression | ((chan[3]) % 20) = partner[3] and ((chan[3]) / 20) = 0 -> (chan[3] % 20) = partner[3] and (chan[3] / 20) = 0.
        if(!(t_oalert_2_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_2() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_3() {
        // SLCO assignment | chan[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        c.chan[3] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_oconnected_0_s_4() {
        // SLCO assignment | chan[partner[3]] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[c.partner[3]], 1);
        c.chan[c.partner[3]] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_2() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_3() {
        // SLCO assignment | chan[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        c.chan[3] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_dveoringout_0_s_4() {
        // SLCO assignment | partner[3] := ((partner[3] % 20) + 0 * 20).
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        c.partner[3] = (((Math.floorMod(c.partner[3], 20)) + 0 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_2() {
        // SLCO assignment | chan[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        c.chan[3] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_3() {
        // SLCO assignment | partner[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        c.partner[3] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_unobtainable_0_s_4() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_0() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_1() {
        // (Superfluous) SLCO expression | true.
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_2() {
        // SLCO assignment | chan[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        c.chan[3] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_3() {
        // SLCO assignment | partner[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        c.partner[3] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_0_s_4() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    // SLCO expression wrapper | record[3] != 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_1_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.record[3], 1);
        return c.record[3] != 255;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_1_s_0() {
        // SLCO expression | record[3] != 255.
        if(!(t_ringback_1_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_ringback_1_s_1() {
        // SLCO assignment | partner[3] := record[3].
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume Perm(c.record[3], 1);
        c.partner[3] = (c.record[3]) & 0xff;
    }

    // SLCO expression wrapper | dev != 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0_n_0() {
        return dev != 1;
    }

    // SLCO expression wrapper | chan[3] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0_n_1() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        return c.chan[3] == 255;
    }

    // SLCO expression wrapper | dev != 1 or chan[3] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0_n_2() {
        return t_talert_0_s_0_n_0() || t_talert_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_0_s_0() {
        // SLCO expression | dev != 1 or chan[3] = 255.
        if(!(t_talert_0_s_0_n_2())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[3]] % 20) = 3.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_1_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[c.partner[3]], 1);
        return (Math.floorMod(c.chan[c.partner[3]], 20)) == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_1_s_0() {
        // SLCO expression | ((chan[partner[3]]) % 20) = 3 -> (chan[partner[3]] % 20) = 3.
        if(!(t_talert_1_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[3]] % 20) != 3.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_2_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[c.partner[3]], 1);
        return (Math.floorMod(c.chan[c.partner[3]], 20)) != 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_talert_2_s_0() {
        // SLCO expression | ((chan[partner[3]]) % 20) != 3 -> (chan[partner[3]] % 20) != 3.
        if(!(t_talert_2_s_0_n_0())) {
            return false;
        }
    }

    // SLCO expression wrapper | (chan[partner[3]] % 20) = 3.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[c.partner[3]], 1);
        return (Math.floorMod(c.chan[c.partner[3]], 20)) == 3;
    }

    // SLCO expression wrapper | (chan[partner[3]] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0_n_1() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[c.partner[3]], 1);
        return (c.chan[c.partner[3]] / 20) == 0;
    }

    // SLCO expression wrapper | (chan[partner[3]] % 20) = 3 and (chan[partner[3]] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0_n_2() {
        return t_tpickup_0_s_0_n_0() && t_tpickup_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_0() {
        // SLCO expression | (chan[partner[3]] % 20) = 3 and (chan[partner[3]] / 20) = 0.
        if(!(t_tpickup_0_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_1() {
        // SLCO assignment | dev := 0.
        dev = (0) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_2() {
        // SLCO assignment | chan[partner[3]] := (3 + 1 * 20).
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[c.partner[3]], 1);
        c.chan[c.partner[3]] = ((3 + 1 * 20)) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_0_s_3() {
        // SLCO assignment | chan[3] := (partner[3] + 1 * 20).
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume Perm(c.chan[3], 1);
        c.chan[3] = ((c.partner[3] + 1 * 20)) & 0xff;
    }

    // SLCO expression wrapper | chan[partner[3]] = 255.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[c.partner[3]], 1);
        return c.chan[c.partner[3]] == 255;
    }

    // SLCO expression wrapper | (chan[partner[3]] % 20) != 3.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0_n_1() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        //@ assume Perm(c.chan[c.partner[3]], 1);
        return (Math.floorMod(c.chan[c.partner[3]], 20)) != 3;
    }

    // SLCO expression wrapper | chan[partner[3]] = 255 or (chan[partner[3]] % 20) != 3.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0_n_2() {
        return t_tpickup_1_s_0_n_0() || t_tpickup_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_0() {
        // SLCO expression | chan[partner[3]] = 255 or (chan[partner[3]] % 20) != 3.
        if(!(t_tpickup_1_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_1() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_2() {
        // SLCO assignment | partner[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        c.partner[3] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tpickup_1_s_3() {
        // SLCO assignment | chan[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        c.chan[3] = (255) & 0xff;
    }

    // SLCO expression wrapper | (chan[3] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        return (c.chan[3] / 20) == 1;
    }

    // SLCO expression wrapper | dev = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0_n_1() {
        return dev == 0;
    }

    // SLCO expression wrapper | (chan[3] / 20) = 1 and dev = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0_n_2() {
        return t_tconnected_0_s_0_n_0() && t_tconnected_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_0() {
        // SLCO expression | (chan[3] / 20) = 1 and dev = 0.
        if(!(t_tconnected_0_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_0_s_1() {
        // SLCO assignment | dev := 1.
        dev = (1) & 0xff;
    }

    // SLCO expression wrapper | (chan[3] / 20) = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        return (c.chan[3] / 20) == 1;
    }

    // SLCO expression wrapper | dev = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0_n_1() {
        return dev == 1;
    }

    // SLCO expression wrapper | (chan[3] / 20) = 1 and dev = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0_n_2() {
        return t_tconnected_1_s_0_n_0() && t_tconnected_1_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_0() {
        // SLCO expression | (chan[3] / 20) = 1 and dev = 1.
        if(!(t_tconnected_1_s_0_n_2())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_1_s_1() {
        // SLCO assignment | dev := 0.
        dev = (0) & 0xff;
    }

    // SLCO expression wrapper | (chan[3] / 20) = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_0_n_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        return (c.chan[3] / 20) == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_0() {
        // SLCO expression | (chan[3] / 20) = 0.
        if(!(t_tconnected_2_s_0_n_0())) {
            return false;
        }
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_1() {
        // SLCO assignment | partner[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.partner[3], 1);
        c.partner[3] = (255) & 0xff;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;
    @*/
    private boolean t_tconnected_2_s_2() {
        // SLCO assignment | chan[3] := 255.
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume Perm(c.chan[3], 1);
        c.chan[3] = (255) & 0xff;
    }
}

// <<< STATE_MACHINE.END (User_3)

// << CLASS.END (GlobalClass)

// < MODEL.END (Telephony)