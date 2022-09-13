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

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_idle_0_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 0; // Lock chan[0]
        dev = 0;
        //@ assert _index == 0;
    }

    

    

    

    

    

    

    

    

    

    

    

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_4_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 0; // Lock partner[0]
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        c.record[c.partner[0]] = 0;
        //@ assert _index == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_4_s_2_lock_rewrite_check_1() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ ghost int _index = c.partner[0]; // Lock callforwardbusy[partner[0]]
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        c.record[c.partner[0]] = 0;
        //@ assume 0 <= 0 && 0 < 4;
        //@ assert _index == c.partner[0];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_2_lock_rewrite_check_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ ghost int _index = c.partner[0]; // Lock chan[partner[0]]
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        c.record[c.partner[0]] = 0;
        //@ assume 0 <= 0 && 0 < 4;
        //@ assert _index == c.partner[0];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_2_lock_rewrite_check_1() {
        //@ ghost int _index = 0; // Lock partner[0]
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        c.record[c.partner[0]] = 0;
        //@ assert _index == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_3_lock_rewrite_check_2() {
        //@ ghost int _index = 0; // Lock chan[0]
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        c.record[c.partner[0]] = 0;
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        c.chan[c.partner[0]] = (0 + 0 * 20);
        //@ assert _index == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_3_lock_rewrite_check_3() {
        //@ ghost int _index = 0; // Lock partner[0]
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        c.record[c.partner[0]] = 0;
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        c.chan[c.partner[0]] = (0 + 0 * 20);
        //@ assert _index == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_busy_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = 0; // Lock partner[0]
        //@ assume 0 <= 0 && 0 < 4;
        c.chan[0] = 255;
        //@ assert _index == 0;
    }

    

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_oconnected_0_s_4_lock_rewrite_check_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ ghost int _index = c.partner[0]; // Lock chan[partner[0]]
        //@ assume 0 <= 0 && 0 < 4;
        c.chan[0] = 255;
        //@ assume 0 <= 0 && 0 < 4;
        //@ assert _index == c.partner[0];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_oconnected_0_s_4_lock_rewrite_check_1() {
        //@ ghost int _index = 0; // Lock partner[0]
        //@ assume 0 <= 0 && 0 < 4;
        c.chan[0] = 255;
        //@ assert _index == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_dveoringout_0_s_4_lock_rewrite_check_0() {
        //@ ghost int _index = 0; // Lock partner[0]
        //@ assume 0 <= 0 && 0 < 4;
        c.chan[0] = 255;
        //@ assert _index == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_unobtainable_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = 0; // Lock partner[0]
        //@ assume 0 <= 0 && 0 < 4;
        c.chan[0] = 255;
        //@ assert _index == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_ringback_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = 0; // Lock partner[0]
        //@ assume 0 <= 0 && 0 < 4;
        c.chan[0] = 255;
        //@ assert _index == 0;
    }

    

    

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_2_lock_rewrite_check_0() {
        //@ assume 0 <= 0 && 0 < 4;
        //@ ghost int _index = c.partner[0]; // Lock chan[partner[0]]
        dev = 0;
        //@ assume 0 <= 0 && 0 < 4;
        //@ assert _index == c.partner[0];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_2_lock_rewrite_check_1() {
        //@ ghost int _index = 0; // Lock partner[0]
        dev = 0;
        //@ assert _index == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_3_lock_rewrite_check_2() {
        //@ ghost int _index = 0; // Lock chan[0]
        dev = 0;
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        c.chan[c.partner[0]] = (0 + 1 * 20);
        //@ assert _index == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_3_lock_rewrite_check_3() {
        //@ ghost int _index = 0; // Lock partner[0]
        dev = 0;
        //@ assume 0 <= 0 && 0 < 4;
        //@ assume 0 <= c.partner[0] && c.partner[0] < 4;
        c.chan[c.partner[0]] = (0 + 1 * 20);
        //@ assert _index == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_1_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 0; // Lock partner[0]
        dev = 1;
        //@ assert _index == 0;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_1_s_3_lock_rewrite_check_1() {
        //@ ghost int _index = 0; // Lock chan[0]
        dev = 1;
        //@ assume 0 <= 0 && 0 < 4;
        c.partner[0] = 255;
        //@ assert _index == 0;
    }

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tconnected_2_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 0; // Lock chan[0]
        //@ assume 0 <= 0 && 0 < 4;
        c.partner[0] = 255;
        //@ assert _index == 0;
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

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_idle_0_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 1; // Lock chan[1]
        dev = 0;
        //@ assert _index == 1;
    }

    

    

    

    

    

    

    

    

    

    

    

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_4_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 1; // Lock partner[1]
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        c.record[c.partner[1]] = 1;
        //@ assert _index == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_4_s_2_lock_rewrite_check_1() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ ghost int _index = c.partner[1]; // Lock callforwardbusy[partner[1]]
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        c.record[c.partner[1]] = 1;
        //@ assume 0 <= 1 && 1 < 4;
        //@ assert _index == c.partner[1];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_2_lock_rewrite_check_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ ghost int _index = c.partner[1]; // Lock chan[partner[1]]
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        c.record[c.partner[1]] = 1;
        //@ assume 0 <= 1 && 1 < 4;
        //@ assert _index == c.partner[1];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_2_lock_rewrite_check_1() {
        //@ ghost int _index = 1; // Lock partner[1]
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        c.record[c.partner[1]] = 1;
        //@ assert _index == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_3_lock_rewrite_check_2() {
        //@ ghost int _index = 1; // Lock chan[1]
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        c.record[c.partner[1]] = 1;
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        c.chan[c.partner[1]] = (1 + 0 * 20);
        //@ assert _index == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_3_lock_rewrite_check_3() {
        //@ ghost int _index = 1; // Lock partner[1]
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        c.record[c.partner[1]] = 1;
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        c.chan[c.partner[1]] = (1 + 0 * 20);
        //@ assert _index == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_busy_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = 1; // Lock partner[1]
        //@ assume 0 <= 1 && 1 < 4;
        c.chan[1] = 255;
        //@ assert _index == 1;
    }

    

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_oconnected_0_s_4_lock_rewrite_check_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ ghost int _index = c.partner[1]; // Lock chan[partner[1]]
        //@ assume 0 <= 1 && 1 < 4;
        c.chan[1] = 255;
        //@ assume 0 <= 1 && 1 < 4;
        //@ assert _index == c.partner[1];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_oconnected_0_s_4_lock_rewrite_check_1() {
        //@ ghost int _index = 1; // Lock partner[1]
        //@ assume 0 <= 1 && 1 < 4;
        c.chan[1] = 255;
        //@ assert _index == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_dveoringout_0_s_4_lock_rewrite_check_0() {
        //@ ghost int _index = 1; // Lock partner[1]
        //@ assume 0 <= 1 && 1 < 4;
        c.chan[1] = 255;
        //@ assert _index == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_unobtainable_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = 1; // Lock partner[1]
        //@ assume 0 <= 1 && 1 < 4;
        c.chan[1] = 255;
        //@ assert _index == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_ringback_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = 1; // Lock partner[1]
        //@ assume 0 <= 1 && 1 < 4;
        c.chan[1] = 255;
        //@ assert _index == 1;
    }

    

    

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_2_lock_rewrite_check_0() {
        //@ assume 0 <= 1 && 1 < 4;
        //@ ghost int _index = c.partner[1]; // Lock chan[partner[1]]
        dev = 0;
        //@ assume 0 <= 1 && 1 < 4;
        //@ assert _index == c.partner[1];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_2_lock_rewrite_check_1() {
        //@ ghost int _index = 1; // Lock partner[1]
        dev = 0;
        //@ assert _index == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_3_lock_rewrite_check_2() {
        //@ ghost int _index = 1; // Lock chan[1]
        dev = 0;
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        c.chan[c.partner[1]] = (1 + 1 * 20);
        //@ assert _index == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_3_lock_rewrite_check_3() {
        //@ ghost int _index = 1; // Lock partner[1]
        dev = 0;
        //@ assume 0 <= 1 && 1 < 4;
        //@ assume 0 <= c.partner[1] && c.partner[1] < 4;
        c.chan[c.partner[1]] = (1 + 1 * 20);
        //@ assert _index == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_1_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 1; // Lock partner[1]
        dev = 1;
        //@ assert _index == 1;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_1_s_3_lock_rewrite_check_1() {
        //@ ghost int _index = 1; // Lock chan[1]
        dev = 1;
        //@ assume 0 <= 1 && 1 < 4;
        c.partner[1] = 255;
        //@ assert _index == 1;
    }

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tconnected_2_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 1; // Lock chan[1]
        //@ assume 0 <= 1 && 1 < 4;
        c.partner[1] = 255;
        //@ assert _index == 1;
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

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_idle_0_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 2; // Lock chan[2]
        dev = 0;
        //@ assert _index == 2;
    }

    

    

    

    

    

    

    

    

    

    

    

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_4_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 2; // Lock partner[2]
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        c.record[c.partner[2]] = 2;
        //@ assert _index == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_4_s_2_lock_rewrite_check_1() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ ghost int _index = c.partner[2]; // Lock callforwardbusy[partner[2]]
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        c.record[c.partner[2]] = 2;
        //@ assume 0 <= 2 && 2 < 4;
        //@ assert _index == c.partner[2];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_2_lock_rewrite_check_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ ghost int _index = c.partner[2]; // Lock chan[partner[2]]
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        c.record[c.partner[2]] = 2;
        //@ assume 0 <= 2 && 2 < 4;
        //@ assert _index == c.partner[2];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_2_lock_rewrite_check_1() {
        //@ ghost int _index = 2; // Lock partner[2]
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        c.record[c.partner[2]] = 2;
        //@ assert _index == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_3_lock_rewrite_check_2() {
        //@ ghost int _index = 2; // Lock chan[2]
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        c.record[c.partner[2]] = 2;
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        c.chan[c.partner[2]] = (2 + 0 * 20);
        //@ assert _index == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_3_lock_rewrite_check_3() {
        //@ ghost int _index = 2; // Lock partner[2]
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        c.record[c.partner[2]] = 2;
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        c.chan[c.partner[2]] = (2 + 0 * 20);
        //@ assert _index == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_busy_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = 2; // Lock partner[2]
        //@ assume 0 <= 2 && 2 < 4;
        c.chan[2] = 255;
        //@ assert _index == 2;
    }

    

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_oconnected_0_s_4_lock_rewrite_check_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ ghost int _index = c.partner[2]; // Lock chan[partner[2]]
        //@ assume 0 <= 2 && 2 < 4;
        c.chan[2] = 255;
        //@ assume 0 <= 2 && 2 < 4;
        //@ assert _index == c.partner[2];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_oconnected_0_s_4_lock_rewrite_check_1() {
        //@ ghost int _index = 2; // Lock partner[2]
        //@ assume 0 <= 2 && 2 < 4;
        c.chan[2] = 255;
        //@ assert _index == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_dveoringout_0_s_4_lock_rewrite_check_0() {
        //@ ghost int _index = 2; // Lock partner[2]
        //@ assume 0 <= 2 && 2 < 4;
        c.chan[2] = 255;
        //@ assert _index == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_unobtainable_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = 2; // Lock partner[2]
        //@ assume 0 <= 2 && 2 < 4;
        c.chan[2] = 255;
        //@ assert _index == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_ringback_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = 2; // Lock partner[2]
        //@ assume 0 <= 2 && 2 < 4;
        c.chan[2] = 255;
        //@ assert _index == 2;
    }

    

    

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_2_lock_rewrite_check_0() {
        //@ assume 0 <= 2 && 2 < 4;
        //@ ghost int _index = c.partner[2]; // Lock chan[partner[2]]
        dev = 0;
        //@ assume 0 <= 2 && 2 < 4;
        //@ assert _index == c.partner[2];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_2_lock_rewrite_check_1() {
        //@ ghost int _index = 2; // Lock partner[2]
        dev = 0;
        //@ assert _index == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_3_lock_rewrite_check_2() {
        //@ ghost int _index = 2; // Lock chan[2]
        dev = 0;
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        c.chan[c.partner[2]] = (2 + 1 * 20);
        //@ assert _index == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_3_lock_rewrite_check_3() {
        //@ ghost int _index = 2; // Lock partner[2]
        dev = 0;
        //@ assume 0 <= 2 && 2 < 4;
        //@ assume 0 <= c.partner[2] && c.partner[2] < 4;
        c.chan[c.partner[2]] = (2 + 1 * 20);
        //@ assert _index == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_1_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 2; // Lock partner[2]
        dev = 1;
        //@ assert _index == 2;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_1_s_3_lock_rewrite_check_1() {
        //@ ghost int _index = 2; // Lock chan[2]
        dev = 1;
        //@ assume 0 <= 2 && 2 < 4;
        c.partner[2] = 255;
        //@ assert _index == 2;
    }

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tconnected_2_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 2; // Lock chan[2]
        //@ assume 0 <= 2 && 2 < 4;
        c.partner[2] = 255;
        //@ assert _index == 2;
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

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_idle_0_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 3; // Lock chan[3]
        dev = 0;
        //@ assert _index == 3;
    }

    

    

    

    

    

    

    

    

    

    

    

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_4_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 3; // Lock partner[3]
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        c.record[c.partner[3]] = 3;
        //@ assert _index == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_4_s_2_lock_rewrite_check_1() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ ghost int _index = c.partner[3]; // Lock callforwardbusy[partner[3]]
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        c.record[c.partner[3]] = 3;
        //@ assume 0 <= 3 && 3 < 4;
        //@ assert _index == c.partner[3];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_2_lock_rewrite_check_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ ghost int _index = c.partner[3]; // Lock chan[partner[3]]
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        c.record[c.partner[3]] = 3;
        //@ assume 0 <= 3 && 3 < 4;
        //@ assert _index == c.partner[3];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_2_lock_rewrite_check_1() {
        //@ ghost int _index = 3; // Lock partner[3]
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        c.record[c.partner[3]] = 3;
        //@ assert _index == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_3_lock_rewrite_check_2() {
        //@ ghost int _index = 3; // Lock chan[3]
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        c.record[c.partner[3]] = 3;
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        c.chan[c.partner[3]] = (3 + 0 * 20);
        //@ assert _index == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_calling_5_s_3_lock_rewrite_check_3() {
        //@ ghost int _index = 3; // Lock partner[3]
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        c.record[c.partner[3]] = 3;
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        c.chan[c.partner[3]] = (3 + 0 * 20);
        //@ assert _index == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_busy_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = 3; // Lock partner[3]
        //@ assume 0 <= 3 && 3 < 4;
        c.chan[3] = 255;
        //@ assert _index == 3;
    }

    

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_oconnected_0_s_4_lock_rewrite_check_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ ghost int _index = c.partner[3]; // Lock chan[partner[3]]
        //@ assume 0 <= 3 && 3 < 4;
        c.chan[3] = 255;
        //@ assume 0 <= 3 && 3 < 4;
        //@ assert _index == c.partner[3];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_oconnected_0_s_4_lock_rewrite_check_1() {
        //@ ghost int _index = 3; // Lock partner[3]
        //@ assume 0 <= 3 && 3 < 4;
        c.chan[3] = 255;
        //@ assert _index == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_dveoringout_0_s_4_lock_rewrite_check_0() {
        //@ ghost int _index = 3; // Lock partner[3]
        //@ assume 0 <= 3 && 3 < 4;
        c.chan[3] = 255;
        //@ assert _index == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_unobtainable_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = 3; // Lock partner[3]
        //@ assume 0 <= 3 && 3 < 4;
        c.chan[3] = 255;
        //@ assert _index == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_ringback_0_s_3_lock_rewrite_check_0() {
        //@ ghost int _index = 3; // Lock partner[3]
        //@ assume 0 <= 3 && 3 < 4;
        c.chan[3] = 255;
        //@ assert _index == 3;
    }

    

    

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_2_lock_rewrite_check_0() {
        //@ assume 0 <= 3 && 3 < 4;
        //@ ghost int _index = c.partner[3]; // Lock chan[partner[3]]
        dev = 0;
        //@ assume 0 <= 3 && 3 < 4;
        //@ assert _index == c.partner[3];
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_2_lock_rewrite_check_1() {
        //@ ghost int _index = 3; // Lock partner[3]
        dev = 0;
        //@ assert _index == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_3_lock_rewrite_check_2() {
        //@ ghost int _index = 3; // Lock chan[3]
        dev = 0;
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        c.chan[c.partner[3]] = (3 + 1 * 20);
        //@ assert _index == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_0_s_3_lock_rewrite_check_3() {
        //@ ghost int _index = 3; // Lock partner[3]
        dev = 0;
        //@ assume 0 <= 3 && 3 < 4;
        //@ assume 0 <= c.partner[3] && c.partner[3] < 4;
        c.chan[c.partner[3]] = (3 + 1 * 20);
        //@ assert _index == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_1_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 3; // Lock partner[3]
        dev = 1;
        //@ assert _index == 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tpickup_1_s_3_lock_rewrite_check_1() {
        //@ ghost int _index = 3; // Lock chan[3]
        dev = 1;
        //@ assume 0 <= 3 && 3 < 4;
        c.partner[3] = 255;
        //@ assert _index == 3;
    }

    

    

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(dev, 1);
    context Perm(mbit, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.chan, 1);
    context Perm(c.partner, 1);
    context Perm(c.callforwardbusy, 1);
    context Perm(c.record, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.chan != null && c.chan.length == 4;
    context c.partner != null && c.partner.length == 4;
    context c.callforwardbusy != null && c.callforwardbusy.length == 4;
    context c.record != null && c.record.length == 4;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.chan[*], 1);
    context Perm(c.partner[*], 1);
    context Perm(c.callforwardbusy[*], 1);
    context Perm(c.record[*], 1);
    @*/
    private boolean t_tconnected_2_s_2_lock_rewrite_check_0() {
        //@ ghost int _index = 3; // Lock chan[3]
        //@ assume 0 <= 3 && 3 < 4;
        c.partner[3] = 255;
        //@ assert _index == 3;
    }
}

// <<< STATE_MACHINE.END (User_3)

// << CLASS.END (GlobalClass)

// < MODEL.END (Telephony)