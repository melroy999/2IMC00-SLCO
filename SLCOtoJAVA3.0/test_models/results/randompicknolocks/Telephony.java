package testing.randompicknolocks;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;
import java.time.Duration;
import java.time.Instant;

// SLCO model Telephony.
public class Telephony {
    // The objects in the model.
    private final SLCO_Class[] objects;

    // Interface for SLCO classes.
    interface SLCO_Class {
        void startThreads();
        void joinThreads();
    }

    Telephony() {
        // Instantiate the objects.
        objects = new SLCO_Class[] {
            new GlobalClass(
                new int[]{ 255, 255, 255, 255 },
                new int[]{ 255, 255, 255, 255 },
                new int[]{ 1, 2, 3, 255 },
                new int[]{ 255, 255, 255, 255 }
            )
        };
    }

    // Lock class to handle locks of global variables
    private static class LockManager {
        // The locks
        private final ReentrantLock[] locks;

        LockManager(int noVariables) {
            locks = new ReentrantLock[noVariables];
            for(int i = 0; i < locks.length; i++) {
                locks[i] = new ReentrantLock(true);
            }
        }

        // Lock method
        void acquire_locks(int[] lock_ids, int end) {
            Arrays.sort(lock_ids, 0, end);
            for (int i = 0; i < end; i++) {
                locks[lock_ids[i]].lock();
            }
        }

        // Unlock method
        void release_locks(int[] lock_ids, int end) {
            for (int i = 0; i < end; i++) {
                locks[lock_ids[i]].unlock();
            }
        }

        // Unlock method during exceptions
        void exception_unlock() {
            System.err.println("Exception encountered. Releasing all locks currently owned by " + Thread.currentThread().getName() + ".");
            for(ReentrantLock lock: locks) {
                while(lock.isHeldByCurrentThread()) {
                    lock.unlock();
                }
            }
        }
    }

    // Representation of the SLCO class GlobalClass.
    private static class GlobalClass implements SLCO_Class {
        // The state machine threads.
        private final Thread T_User_0;
        private final Thread T_User_1;
        private final Thread T_User_2;
        private final Thread T_User_3;

        // Class variables.
        private final int[] chan;
        private final int[] partner;
        private final int[] callforwardbusy;
        private final int[] record;

        GlobalClass(int[] chan, int[] partner, int[] callforwardbusy, int[] record) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(16);

            // Instantiate the class variables.
            this.chan = chan;
            this.partner = partner;
            this.callforwardbusy = callforwardbusy;
            this.record = record;

            // Instantiate the state machine threads and pass on the class' lock manager.
            T_User_0 = new GlobalClass_User_0Thread(lockManager);
            T_User_1 = new GlobalClass_User_1Thread(lockManager);
            T_User_2 = new GlobalClass_User_2Thread(lockManager);
            T_User_3 = new GlobalClass_User_3Thread(lockManager);
        }

        // Define the states fot the state machine User_0.
        interface GlobalClass_User_0Thread_States {
            enum States {
                idle, 
                dialing, 
                calling, 
                busy, 
                qi, 
                talert, 
                unobtainable, 
                oalert, 
                errorstate, 
                oconnected, 
                dveoringout, 
                tpickup, 
                tconnected, 
                ringback
            }
        }

        // Representation of the SLCO state machine User_0.
        class GlobalClass_User_0Thread extends Thread implements GlobalClass_User_0Thread_States {
            // Current state
            private GlobalClass_User_0Thread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // Thread local variables.
            private int dev;
            private int mbit;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            GlobalClass_User_0Thread(LockManager lockManagerInstance) {
                currentState = GlobalClass_User_0Thread.States.idle;
                lockManager = lockManagerInstance;
                lock_ids = new int[0];
                target_locks = new int[0];
                random = new Random();

                // Variable instantiations.
                dev = (char) 1;
                mbit = (char) 0;
            }

            // SLCO transition (p:0, id:0) | idle -> dialing | [chan[0] = 255; dev := 0; chan[0] := (0 + 0 * 20)].
            private boolean execute_transition_idle_0() {
                // SLCO composite | [chan[0] = 255; dev := 0; chan[0] := ((0) + (0) * 20)] -> [chan[0] = 255; dev := 0; chan[0] := (0 + 0 * 20)].
                // SLCO expression | chan[0] = 255.
                if(!(chan[0] == 255)) {
                    return false;
                }
                // SLCO assignment | dev := 0.
                dev = (0) & 0xff;
                // SLCO assignment | chan[0] := (0 + 0 * 20).
                chan[0] = ((0 + 0 * 20)) & 0xff;

                currentState = GlobalClass_User_0Thread.States.dialing;
                return true;
            }

            // SLCO transition (p:0, id:1) | idle -> qi | [chan[0] != 255; partner[0] := (chan[0] % 20)].
            private boolean execute_transition_idle_1() {
                // SLCO composite | [chan[0] != 255; partner[0] := ((chan[0]) % 20)] -> [chan[0] != 255; partner[0] := (chan[0] % 20)].
                // SLCO expression | chan[0] != 255.
                if(!(chan[0] != 255)) {
                    return false;
                }
                // SLCO assignment | partner[0] := (chan[0] % 20).
                partner[0] = ((Math.floorMod(chan[0], 20))) & 0xff;

                currentState = GlobalClass_User_0Thread.States.qi;
                return true;
            }

            // SLCO transition (p:0, id:0) | qi -> talert | (chan[partner[0]] % 20) = 0.
            private boolean execute_transition_qi_0() {
                // SLCO expression | ((chan[partner[0]]) % 20) = 0 -> (chan[partner[0]] % 20) = 0.
                if(!((Math.floorMod(chan[partner[0]], 20)) == 0)) {
                    return false;
                }

                currentState = GlobalClass_User_0Thread.States.talert;
                return true;
            }

            // SLCO transition (p:0, id:1) | qi -> idle | [(chan[partner[0]] % 20) != 0; partner[0] := 255].
            private boolean execute_transition_qi_1() {
                // SLCO composite | [((chan[partner[0]]) % 20) != 0; partner[0] := 255] -> [(chan[partner[0]] % 20) != 0; partner[0] := 255].
                // SLCO expression | (chan[partner[0]] % 20) != 0.
                if(!((Math.floorMod(chan[partner[0]], 20)) != 0)) {
                    return false;
                }
                // SLCO assignment | partner[0] := 255.
                partner[0] = (255) & 0xff;

                currentState = GlobalClass_User_0Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | dialing -> idle | true | [true; dev := 1; chan[0] := 255].
            private boolean execute_transition_dialing_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [dev := 1; chan[0] := 255] -> [true; dev := 1; chan[0] := 255].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | chan[0] := 255.
                chan[0] = (255) & 0xff;

                currentState = GlobalClass_User_0Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:1) | dialing -> calling | true | partner[0] := 0.
            private boolean execute_transition_dialing_1() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[0] := 0] -> partner[0] := 0.
                partner[0] = (0) & 0xff;

                currentState = GlobalClass_User_0Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:2) | dialing -> calling | true | partner[0] := 1.
            private boolean execute_transition_dialing_2() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[0] := 1] -> partner[0] := 1.
                partner[0] = (1) & 0xff;

                currentState = GlobalClass_User_0Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:3) | dialing -> calling | true | partner[0] := 2.
            private boolean execute_transition_dialing_3() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[0] := 2] -> partner[0] := 2.
                partner[0] = (2) & 0xff;

                currentState = GlobalClass_User_0Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:4) | dialing -> calling | true | partner[0] := 3.
            private boolean execute_transition_dialing_4() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[0] := 3] -> partner[0] := 3.
                partner[0] = (3) & 0xff;

                currentState = GlobalClass_User_0Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:5) | dialing -> calling | true | partner[0] := 4.
            private boolean execute_transition_dialing_5() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[0] := 4] -> partner[0] := 4.
                partner[0] = (4) & 0xff;

                currentState = GlobalClass_User_0Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:0) | calling -> busy | partner[0] = 0.
            private boolean execute_transition_calling_0() {
                // SLCO expression | partner[0] = 0.
                if(!(partner[0] == 0)) {
                    return false;
                }

                currentState = GlobalClass_User_0Thread.States.busy;
                return true;
            }

            // SLCO transition (p:0, id:1) | calling -> unobtainable | partner[0] = 4.
            private boolean execute_transition_calling_1() {
                // SLCO expression | partner[0] = 4.
                if(!(partner[0] == 4)) {
                    return false;
                }

                currentState = GlobalClass_User_0Thread.States.unobtainable;
                return true;
            }

            // SLCO transition (p:0, id:2) | calling -> ringback | partner[0] = 4.
            private boolean execute_transition_calling_2() {
                // SLCO expression | partner[0] = 4.
                if(!(partner[0] == 4)) {
                    return false;
                }

                currentState = GlobalClass_User_0Thread.States.ringback;
                return true;
            }

            // SLCO transition (p:0, id:3) | calling -> busy | [partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255 and callforwardbusy[partner[0]] = 255; record[partner[0]] := 0].
            private boolean execute_transition_calling_3() {
                // SLCO composite | [partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255 and callforwardbusy[partner[0]] = 255; record[partner[0]] := 0].
                // SLCO expression | partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255 and callforwardbusy[partner[0]] = 255.
                if(!(partner[0] != 0 && partner[0] != 4 && chan[partner[0]] != 255 && callforwardbusy[partner[0]] == 255)) {
                    return false;
                }
                // SLCO assignment | record[partner[0]] := 0.
                record[partner[0]] = (0) & 0xff;

                currentState = GlobalClass_User_0Thread.States.busy;
                return true;
            }

            // SLCO transition (p:0, id:4) | calling -> calling | [partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255 and callforwardbusy[partner[0]] != 255; record[partner[0]] := 0; partner[0] := callforwardbusy[partner[0]]].
            private boolean execute_transition_calling_4() {
                // SLCO composite | [partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255 and callforwardbusy[partner[0]] != 255; record[partner[0]] := 0; partner[0] := callforwardbusy[partner[0]]].
                // SLCO expression | partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255 and callforwardbusy[partner[0]] != 255.
                if(!(partner[0] != 0 && partner[0] != 4 && chan[partner[0]] != 255 && callforwardbusy[partner[0]] != 255)) {
                    return false;
                }
                // SLCO assignment | record[partner[0]] := 0.
                record[partner[0]] = (0) & 0xff;
                // SLCO assignment | partner[0] := callforwardbusy[partner[0]].
                partner[0] = (callforwardbusy[partner[0]]) & 0xff;

                currentState = GlobalClass_User_0Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:5) | calling -> oalert | [partner[0] != 0 and partner[0] != 4 and chan[partner[0]] = 255; record[partner[0]] := 0; chan[partner[0]] := (0 + 0 * 20); chan[0] := (partner[0] + 0 * 20)].
            private boolean execute_transition_calling_5() {
                // SLCO composite | [partner[0] != 0 and partner[0] != 4 and chan[partner[0]] = 255; record[partner[0]] := 0; chan[partner[0]] := ((0) + (0) * 20); chan[0] := ((partner[0]) + (0) * 20)] -> [partner[0] != 0 and partner[0] != 4 and chan[partner[0]] = 255; record[partner[0]] := 0; chan[partner[0]] := (0 + 0 * 20); chan[0] := (partner[0] + 0 * 20)].
                // SLCO expression | partner[0] != 0 and partner[0] != 4 and chan[partner[0]] = 255.
                if(!(partner[0] != 0 && partner[0] != 4 && chan[partner[0]] == 255)) {
                    return false;
                }
                // SLCO assignment | record[partner[0]] := 0.
                record[partner[0]] = (0) & 0xff;
                // SLCO assignment | chan[partner[0]] := (0 + 0 * 20).
                chan[partner[0]] = ((0 + 0 * 20)) & 0xff;
                // SLCO assignment | chan[0] := (partner[0] + 0 * 20).
                chan[0] = ((partner[0] + 0 * 20)) & 0xff;

                currentState = GlobalClass_User_0Thread.States.oalert;
                return true;
            }

            // SLCO transition (p:0, id:0) | busy -> idle | true | [true; chan[0] := 255; partner[0] := 255; dev := 1].
            private boolean execute_transition_busy_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [chan[0] := 255; partner[0] := 255; dev := 1] -> [true; chan[0] := 255; partner[0] := 255; dev := 1].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | chan[0] := 255.
                chan[0] = (255) & 0xff;
                // SLCO assignment | partner[0] := 255.
                partner[0] = (255) & 0xff;
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_0Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | oalert -> errorstate | (chan[0] % 20) != partner[0].
            private boolean execute_transition_oalert_0() {
                // SLCO expression | ((chan[0]) % 20) != partner[0] -> (chan[0] % 20) != partner[0].
                if(!((Math.floorMod(chan[0], 20)) != partner[0])) {
                    return false;
                }

                currentState = GlobalClass_User_0Thread.States.errorstate;
                return true;
            }

            // SLCO transition (p:0, id:1) | oalert -> oconnected | (chan[0] % 20) = partner[0] and (chan[0] / 20) = 1.
            private boolean execute_transition_oalert_1() {
                // SLCO expression | ((chan[0]) % 20) = partner[0] and ((chan[0]) / 20) = 1 -> (chan[0] % 20) = partner[0] and (chan[0] / 20) = 1.
                if(!((Math.floorMod(chan[0], 20)) == partner[0] && (chan[0] / 20) == 1)) {
                    return false;
                }

                currentState = GlobalClass_User_0Thread.States.oconnected;
                return true;
            }

            // SLCO transition (p:0, id:2) | oalert -> dveoringout | (chan[0] % 20) = partner[0] and (chan[0] / 20) = 0.
            private boolean execute_transition_oalert_2() {
                // SLCO expression | ((chan[0]) % 20) = partner[0] and ((chan[0]) / 20) = 0 -> (chan[0] % 20) = partner[0] and (chan[0] / 20) = 0.
                if(!((Math.floorMod(chan[0], 20)) == partner[0] && (chan[0] / 20) == 0)) {
                    return false;
                }

                currentState = GlobalClass_User_0Thread.States.dveoringout;
                return true;
            }

            // SLCO transition (p:0, id:0) | oconnected -> idle | true | [true; dev := 1; chan[0] := 255; chan[partner[0]] := 255].
            private boolean execute_transition_oconnected_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [dev := 1; chan[0] := 255; chan[partner[0]] := 255] -> [true; dev := 1; chan[0] := 255; chan[partner[0]] := 255].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | chan[0] := 255.
                chan[0] = (255) & 0xff;
                // SLCO assignment | chan[partner[0]] := 255.
                chan[partner[0]] = (255) & 0xff;

                currentState = GlobalClass_User_0Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | dveoringout -> idle | true | [true; dev := 1; chan[0] := 255; partner[0] := ((partner[0] % 20) + 0 * 20)].
            private boolean execute_transition_dveoringout_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [dev := 1; chan[0] := 255; partner[0] := ((((partner[0]) % 20)) + (0) * 20)] -> [true; dev := 1; chan[0] := 255; partner[0] := ((partner[0] % 20) + 0 * 20)].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | chan[0] := 255.
                chan[0] = (255) & 0xff;
                // SLCO assignment | partner[0] := ((partner[0] % 20) + 0 * 20).
                partner[0] = (((Math.floorMod(partner[0], 20)) + 0 * 20)) & 0xff;

                currentState = GlobalClass_User_0Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | unobtainable -> idle | true | [true; chan[0] := 255; partner[0] := 255; dev := 1].
            private boolean execute_transition_unobtainable_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [chan[0] := 255; partner[0] := 255; dev := 1] -> [true; chan[0] := 255; partner[0] := 255; dev := 1].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | chan[0] := 255.
                chan[0] = (255) & 0xff;
                // SLCO assignment | partner[0] := 255.
                partner[0] = (255) & 0xff;
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_0Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | ringback -> idle | true | [true; chan[0] := 255; partner[0] := 255; dev := 1].
            private boolean execute_transition_ringback_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [chan[0] := 255; partner[0] := 255; dev := 1] -> [true; chan[0] := 255; partner[0] := 255; dev := 1].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | chan[0] := 255.
                chan[0] = (255) & 0xff;
                // SLCO assignment | partner[0] := 255.
                partner[0] = (255) & 0xff;
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_0Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:1) | ringback -> calling | [record[0] != 255; partner[0] := record[0]].
            private boolean execute_transition_ringback_1() {
                // SLCO composite | [record[0] != 255; partner[0] := record[0]].
                // SLCO expression | record[0] != 255.
                if(!(record[0] != 255)) {
                    return false;
                }
                // SLCO assignment | partner[0] := record[0].
                partner[0] = (record[0]) & 0xff;

                currentState = GlobalClass_User_0Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:0) | talert -> errorstate | dev != 1 or chan[0] = 255.
            private boolean execute_transition_talert_0() {
                // SLCO expression | dev != 1 or chan[0] = 255.
                if(!(dev != 1 || chan[0] == 255)) {
                    return false;
                }

                currentState = GlobalClass_User_0Thread.States.errorstate;
                return true;
            }

            // SLCO transition (p:0, id:1) | talert -> tpickup | (chan[partner[0]] % 20) = 0.
            private boolean execute_transition_talert_1() {
                // SLCO expression | ((chan[partner[0]]) % 20) = 0 -> (chan[partner[0]] % 20) = 0.
                if(!((Math.floorMod(chan[partner[0]], 20)) == 0)) {
                    return false;
                }

                currentState = GlobalClass_User_0Thread.States.tpickup;
                return true;
            }

            // SLCO transition (p:0, id:2) | talert -> idle | (chan[partner[0]] % 20) != 0.
            private boolean execute_transition_talert_2() {
                // SLCO expression | ((chan[partner[0]]) % 20) != 0 -> (chan[partner[0]] % 20) != 0.
                if(!((Math.floorMod(chan[partner[0]], 20)) != 0)) {
                    return false;
                }

                currentState = GlobalClass_User_0Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | tpickup -> tconnected | [(chan[partner[0]] % 20) = 0 and (chan[partner[0]] / 20) = 0; dev := 0; chan[partner[0]] := (0 + 1 * 20); chan[0] := (partner[0] + 1 * 20)].
            private boolean execute_transition_tpickup_0() {
                // SLCO composite | [((chan[partner[0]]) % 20) = 0 and ((chan[partner[0]]) / 20) = 0; dev := 0; chan[partner[0]] := ((0) + (1) * 20); chan[0] := ((partner[0]) + (1) * 20)] -> [(chan[partner[0]] % 20) = 0 and (chan[partner[0]] / 20) = 0; dev := 0; chan[partner[0]] := (0 + 1 * 20); chan[0] := (partner[0] + 1 * 20)].
                // SLCO expression | (chan[partner[0]] % 20) = 0 and (chan[partner[0]] / 20) = 0.
                if(!((Math.floorMod(chan[partner[0]], 20)) == 0 && (chan[partner[0]] / 20) == 0)) {
                    return false;
                }
                // SLCO assignment | dev := 0.
                dev = (0) & 0xff;
                // SLCO assignment | chan[partner[0]] := (0 + 1 * 20).
                chan[partner[0]] = ((0 + 1 * 20)) & 0xff;
                // SLCO assignment | chan[0] := (partner[0] + 1 * 20).
                chan[0] = ((partner[0] + 1 * 20)) & 0xff;

                currentState = GlobalClass_User_0Thread.States.tconnected;
                return true;
            }

            // SLCO transition (p:0, id:1) | tpickup -> idle | [chan[partner[0]] = 255 or (chan[partner[0]] % 20) != 0; dev := 1; partner[0] := 255; chan[0] := 255].
            private boolean execute_transition_tpickup_1() {
                // SLCO composite | [chan[partner[0]] = 255 or ((chan[partner[0]]) % 20) != 0; dev := 1; partner[0] := 255; chan[0] := 255] -> [chan[partner[0]] = 255 or (chan[partner[0]] % 20) != 0; dev := 1; partner[0] := 255; chan[0] := 255].
                // SLCO expression | chan[partner[0]] = 255 or (chan[partner[0]] % 20) != 0.
                if(!(chan[partner[0]] == 255 || (Math.floorMod(chan[partner[0]], 20)) != 0)) {
                    return false;
                }
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | partner[0] := 255.
                partner[0] = (255) & 0xff;
                // SLCO assignment | chan[0] := 255.
                chan[0] = (255) & 0xff;

                currentState = GlobalClass_User_0Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | tconnected -> tconnected | [(chan[0] / 20) = 1 and dev = 0; dev := 1].
            private boolean execute_transition_tconnected_0() {
                // SLCO composite | [((chan[0]) / 20) = 1 and dev = 0; dev := 1] -> [(chan[0] / 20) = 1 and dev = 0; dev := 1].
                // SLCO expression | (chan[0] / 20) = 1 and dev = 0.
                if(!((chan[0] / 20) == 1 && dev == 0)) {
                    return false;
                }
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_0Thread.States.tconnected;
                return true;
            }

            // SLCO transition (p:0, id:1) | tconnected -> tconnected | [(chan[0] / 20) = 1 and dev = 1; dev := 0].
            private boolean execute_transition_tconnected_1() {
                // SLCO composite | [((chan[0]) / 20) = 1 and dev = 1; dev := 0] -> [(chan[0] / 20) = 1 and dev = 1; dev := 0].
                // SLCO expression | (chan[0] / 20) = 1 and dev = 1.
                if(!((chan[0] / 20) == 1 && dev == 1)) {
                    return false;
                }
                // SLCO assignment | dev := 0.
                dev = (0) & 0xff;

                currentState = GlobalClass_User_0Thread.States.tconnected;
                return true;
            }

            // SLCO transition (p:0, id:2) | tconnected -> idle | [(chan[0] / 20) = 0; partner[0] := 255; chan[0] := 255].
            private boolean execute_transition_tconnected_2() {
                // SLCO composite | [((chan[0]) / 20) = 0; partner[0] := 255; chan[0] := 255] -> [(chan[0] / 20) = 0; partner[0] := 255; chan[0] := 255].
                // SLCO expression | (chan[0] / 20) = 0.
                if(!((chan[0] / 20) == 0)) {
                    return false;
                }
                // SLCO assignment | partner[0] := 255.
                partner[0] = (255) & 0xff;
                // SLCO assignment | chan[0] := 255.
                chan[0] = (255) & 0xff;

                currentState = GlobalClass_User_0Thread.States.idle;
                return true;
            }

            // Attempt to fire a transition starting in state idle.
            private void exec_idle() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | idle -> dialing | [chan[0] = 255; dev := 0; chan[0] := (0 + 0 * 20)].
                if(execute_transition_idle_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | idle -> qi | [chan[0] != 255; partner[0] := (chan[0] % 20)].
                if(execute_transition_idle_1()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state dialing.
            private void exec_dialing() {
                // [N_DET.START]
                switch(random.nextInt(6)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | dialing -> idle | true | [true; dev := 1; chan[0] := 255].
                        if(execute_transition_dialing_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (p:0, id:1) | dialing -> calling | true | partner[0] := 0.
                        if(execute_transition_dialing_1()) {
                            return;
                        }
                    }
                    case 2 -> {
                        // SLCO transition (p:0, id:2) | dialing -> calling | true | partner[0] := 1.
                        if(execute_transition_dialing_2()) {
                            return;
                        }
                    }
                    case 3 -> {
                        // SLCO transition (p:0, id:3) | dialing -> calling | true | partner[0] := 2.
                        if(execute_transition_dialing_3()) {
                            return;
                        }
                    }
                    case 4 -> {
                        // SLCO transition (p:0, id:4) | dialing -> calling | true | partner[0] := 3.
                        if(execute_transition_dialing_4()) {
                            return;
                        }
                    }
                    case 5 -> {
                        // SLCO transition (p:0, id:5) | dialing -> calling | true | partner[0] := 4.
                        if(execute_transition_dialing_5()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state calling.
            private void exec_calling() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | calling -> busy | partner[0] = 0.
                if(execute_transition_calling_0()) {
                    return;
                }
                // SLCO expression | partner[0] = 4.
                if(partner[0] == 4) {
                    // [N_DET.START]
                    switch(random.nextInt(2)) {
                        case 0 -> {
                            // SLCO transition (p:0, id:1) | calling -> unobtainable | partner[0] = 4.
                            if(execute_transition_calling_1()) {
                                return;
                            }
                        }
                        case 1 -> {
                            // SLCO transition (p:0, id:2) | calling -> ringback | partner[0] = 4.
                            if(execute_transition_calling_2()) {
                                return;
                            }
                        }
                    }
                    // [N_DET.END]
                }
                // SLCO transition (p:0, id:3) | calling -> busy | [partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255 and callforwardbusy[partner[0]] = 255; record[partner[0]] := 0].
                if(execute_transition_calling_3()) {
                    return;
                }
                // SLCO transition (p:0, id:4) | calling -> calling | [partner[0] != 0 and partner[0] != 4 and chan[partner[0]] != 255 and callforwardbusy[partner[0]] != 255; record[partner[0]] := 0; partner[0] := callforwardbusy[partner[0]]].
                if(execute_transition_calling_4()) {
                    return;
                }
                // SLCO transition (p:0, id:5) | calling -> oalert | [partner[0] != 0 and partner[0] != 4 and chan[partner[0]] = 255; record[partner[0]] := 0; chan[partner[0]] := (0 + 0 * 20); chan[0] := (partner[0] + 0 * 20)].
                if(execute_transition_calling_5()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state busy.
            private void exec_busy() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | busy -> idle | true | [true; chan[0] := 255; partner[0] := 255; dev := 1].
                if(execute_transition_busy_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state qi.
            private void exec_qi() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | qi -> talert | (chan[partner[0]] % 20) = 0.
                if(execute_transition_qi_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | qi -> idle | [(chan[partner[0]] % 20) != 0; partner[0] := 255].
                if(execute_transition_qi_1()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state talert.
            private void exec_talert() {
                // [N_DET.START]
                switch(random.nextInt(2)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | talert -> errorstate | dev != 1 or chan[0] = 255.
                        if(execute_transition_talert_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // [DET.START]
                        // SLCO transition (p:0, id:1) | talert -> tpickup | (chan[partner[0]] % 20) = 0.
                        if(execute_transition_talert_1()) {
                            return;
                        }
                        // SLCO transition (p:0, id:2) | talert -> idle | (chan[partner[0]] % 20) != 0.
                        if(execute_transition_talert_2()) {
                            return;
                        }
                        // [DET.END]
                    }
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state unobtainable.
            private void exec_unobtainable() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | unobtainable -> idle | true | [true; chan[0] := 255; partner[0] := 255; dev := 1].
                if(execute_transition_unobtainable_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state oalert.
            private void exec_oalert() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | oalert -> errorstate | (chan[0] % 20) != partner[0].
                if(execute_transition_oalert_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | oalert -> oconnected | (chan[0] % 20) = partner[0] and (chan[0] / 20) = 1.
                if(execute_transition_oalert_1()) {
                    return;
                }
                // SLCO transition (p:0, id:2) | oalert -> dveoringout | (chan[0] % 20) = partner[0] and (chan[0] / 20) = 0.
                if(execute_transition_oalert_2()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state errorstate.
            private void exec_errorstate() {
                // There are no transitions starting in state errorstate.
            }

            // Attempt to fire a transition starting in state oconnected.
            private void exec_oconnected() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | oconnected -> idle | true | [true; dev := 1; chan[0] := 255; chan[partner[0]] := 255].
                if(execute_transition_oconnected_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state dveoringout.
            private void exec_dveoringout() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | dveoringout -> idle | true | [true; dev := 1; chan[0] := 255; partner[0] := ((partner[0] % 20) + 0 * 20)].
                if(execute_transition_dveoringout_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state tpickup.
            private void exec_tpickup() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | tpickup -> tconnected | [(chan[partner[0]] % 20) = 0 and (chan[partner[0]] / 20) = 0; dev := 0; chan[partner[0]] := (0 + 1 * 20); chan[0] := (partner[0] + 1 * 20)].
                if(execute_transition_tpickup_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | tpickup -> idle | [chan[partner[0]] = 255 or (chan[partner[0]] % 20) != 0; dev := 1; partner[0] := 255; chan[0] := 255].
                if(execute_transition_tpickup_1()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state tconnected.
            private void exec_tconnected() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | tconnected -> tconnected | [(chan[0] / 20) = 1 and dev = 0; dev := 1].
                if(execute_transition_tconnected_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | tconnected -> tconnected | [(chan[0] / 20) = 1 and dev = 1; dev := 0].
                if(execute_transition_tconnected_1()) {
                    return;
                }
                // SLCO transition (p:0, id:2) | tconnected -> idle | [(chan[0] / 20) = 0; partner[0] := 255; chan[0] := 255].
                if(execute_transition_tconnected_2()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state ringback.
            private void exec_ringback() {
                // [N_DET.START]
                switch(random.nextInt(2)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | ringback -> idle | true | [true; chan[0] := 255; partner[0] := 255; dev := 1].
                        if(execute_transition_ringback_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (p:0, id:1) | ringback -> calling | [record[0] != 255; partner[0] := record[0]].
                        if(execute_transition_ringback_1()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 30) {
                    switch(currentState) {
                        case idle -> exec_idle();
                        case dialing -> exec_dialing();
                        case calling -> exec_calling();
                        case busy -> exec_busy();
                        case qi -> exec_qi();
                        case talert -> exec_talert();
                        case unobtainable -> exec_unobtainable();
                        case oalert -> exec_oalert();
                        case errorstate -> exec_errorstate();
                        case oconnected -> exec_oconnected();
                        case dveoringout -> exec_dveoringout();
                        case tpickup -> exec_tpickup();
                        case tconnected -> exec_tconnected();
                        case ringback -> exec_ringback();
                    }
                }
            }

            // The thread's run method.
            public void run() {
                try {
                    exec();
                } catch(Exception e) {
                    lockManager.exception_unlock();
                    throw e;
                }
            }
        }

        // Define the states fot the state machine User_1.
        interface GlobalClass_User_1Thread_States {
            enum States {
                idle, 
                dialing, 
                calling, 
                busy, 
                qi, 
                talert, 
                unobtainable, 
                oalert, 
                errorstate, 
                oconnected, 
                dveoringout, 
                tpickup, 
                tconnected, 
                ringback
            }
        }

        // Representation of the SLCO state machine User_1.
        class GlobalClass_User_1Thread extends Thread implements GlobalClass_User_1Thread_States {
            // Current state
            private GlobalClass_User_1Thread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // Thread local variables.
            private int dev;
            private int mbit;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            GlobalClass_User_1Thread(LockManager lockManagerInstance) {
                currentState = GlobalClass_User_1Thread.States.idle;
                lockManager = lockManagerInstance;
                lock_ids = new int[0];
                target_locks = new int[0];
                random = new Random();

                // Variable instantiations.
                dev = (char) 1;
                mbit = (char) 0;
            }

            // SLCO transition (p:0, id:0) | idle -> dialing | [chan[1] = 255; dev := 0; chan[1] := (1 + 0 * 20)].
            private boolean execute_transition_idle_0() {
                // SLCO composite | [chan[1] = 255; dev := 0; chan[1] := ((1) + (0) * 20)] -> [chan[1] = 255; dev := 0; chan[1] := (1 + 0 * 20)].
                // SLCO expression | chan[1] = 255.
                if(!(chan[1] == 255)) {
                    return false;
                }
                // SLCO assignment | dev := 0.
                dev = (0) & 0xff;
                // SLCO assignment | chan[1] := (1 + 0 * 20).
                chan[1] = ((1 + 0 * 20)) & 0xff;

                currentState = GlobalClass_User_1Thread.States.dialing;
                return true;
            }

            // SLCO transition (p:0, id:1) | idle -> qi | [chan[1] != 255; partner[1] := (chan[1] % 20)].
            private boolean execute_transition_idle_1() {
                // SLCO composite | [chan[1] != 255; partner[1] := ((chan[1]) % 20)] -> [chan[1] != 255; partner[1] := (chan[1] % 20)].
                // SLCO expression | chan[1] != 255.
                if(!(chan[1] != 255)) {
                    return false;
                }
                // SLCO assignment | partner[1] := (chan[1] % 20).
                partner[1] = ((Math.floorMod(chan[1], 20))) & 0xff;

                currentState = GlobalClass_User_1Thread.States.qi;
                return true;
            }

            // SLCO transition (p:0, id:0) | qi -> talert | (chan[partner[1]] % 20) = 1.
            private boolean execute_transition_qi_0() {
                // SLCO expression | ((chan[partner[1]]) % 20) = 1 -> (chan[partner[1]] % 20) = 1.
                if(!((Math.floorMod(chan[partner[1]], 20)) == 1)) {
                    return false;
                }

                currentState = GlobalClass_User_1Thread.States.talert;
                return true;
            }

            // SLCO transition (p:0, id:1) | qi -> idle | [(chan[partner[1]] % 20) != 1; partner[1] := 255].
            private boolean execute_transition_qi_1() {
                // SLCO composite | [((chan[partner[1]]) % 20) != 1; partner[1] := 255] -> [(chan[partner[1]] % 20) != 1; partner[1] := 255].
                // SLCO expression | (chan[partner[1]] % 20) != 1.
                if(!((Math.floorMod(chan[partner[1]], 20)) != 1)) {
                    return false;
                }
                // SLCO assignment | partner[1] := 255.
                partner[1] = (255) & 0xff;

                currentState = GlobalClass_User_1Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | dialing -> idle | true | [true; dev := 1; chan[1] := 255].
            private boolean execute_transition_dialing_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [dev := 1; chan[1] := 255] -> [true; dev := 1; chan[1] := 255].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | chan[1] := 255.
                chan[1] = (255) & 0xff;

                currentState = GlobalClass_User_1Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:1) | dialing -> calling | true | partner[1] := 0.
            private boolean execute_transition_dialing_1() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[1] := 0] -> partner[1] := 0.
                partner[1] = (0) & 0xff;

                currentState = GlobalClass_User_1Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:2) | dialing -> calling | true | partner[1] := 1.
            private boolean execute_transition_dialing_2() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[1] := 1] -> partner[1] := 1.
                partner[1] = (1) & 0xff;

                currentState = GlobalClass_User_1Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:3) | dialing -> calling | true | partner[1] := 2.
            private boolean execute_transition_dialing_3() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[1] := 2] -> partner[1] := 2.
                partner[1] = (2) & 0xff;

                currentState = GlobalClass_User_1Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:4) | dialing -> calling | true | partner[1] := 3.
            private boolean execute_transition_dialing_4() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[1] := 3] -> partner[1] := 3.
                partner[1] = (3) & 0xff;

                currentState = GlobalClass_User_1Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:5) | dialing -> calling | true | partner[1] := 4.
            private boolean execute_transition_dialing_5() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[1] := 4] -> partner[1] := 4.
                partner[1] = (4) & 0xff;

                currentState = GlobalClass_User_1Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:0) | calling -> busy | partner[1] = 1.
            private boolean execute_transition_calling_0() {
                // SLCO expression | partner[1] = 1.
                if(!(partner[1] == 1)) {
                    return false;
                }

                currentState = GlobalClass_User_1Thread.States.busy;
                return true;
            }

            // SLCO transition (p:0, id:1) | calling -> unobtainable | partner[1] = 4.
            private boolean execute_transition_calling_1() {
                // SLCO expression | partner[1] = 4.
                if(!(partner[1] == 4)) {
                    return false;
                }

                currentState = GlobalClass_User_1Thread.States.unobtainable;
                return true;
            }

            // SLCO transition (p:0, id:2) | calling -> ringback | partner[1] = 4.
            private boolean execute_transition_calling_2() {
                // SLCO expression | partner[1] = 4.
                if(!(partner[1] == 4)) {
                    return false;
                }

                currentState = GlobalClass_User_1Thread.States.ringback;
                return true;
            }

            // SLCO transition (p:0, id:3) | calling -> busy | [partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255 and callforwardbusy[partner[1]] = 255; record[partner[1]] := 1].
            private boolean execute_transition_calling_3() {
                // SLCO composite | [partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255 and callforwardbusy[partner[1]] = 255; record[partner[1]] := 1].
                // SLCO expression | partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255 and callforwardbusy[partner[1]] = 255.
                if(!(partner[1] != 1 && partner[1] != 4 && chan[partner[1]] != 255 && callforwardbusy[partner[1]] == 255)) {
                    return false;
                }
                // SLCO assignment | record[partner[1]] := 1.
                record[partner[1]] = (1) & 0xff;

                currentState = GlobalClass_User_1Thread.States.busy;
                return true;
            }

            // SLCO transition (p:0, id:4) | calling -> calling | [partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255 and callforwardbusy[partner[1]] != 255; record[partner[1]] := 1; partner[1] := callforwardbusy[partner[1]]].
            private boolean execute_transition_calling_4() {
                // SLCO composite | [partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255 and callforwardbusy[partner[1]] != 255; record[partner[1]] := 1; partner[1] := callforwardbusy[partner[1]]].
                // SLCO expression | partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255 and callforwardbusy[partner[1]] != 255.
                if(!(partner[1] != 1 && partner[1] != 4 && chan[partner[1]] != 255 && callforwardbusy[partner[1]] != 255)) {
                    return false;
                }
                // SLCO assignment | record[partner[1]] := 1.
                record[partner[1]] = (1) & 0xff;
                // SLCO assignment | partner[1] := callforwardbusy[partner[1]].
                partner[1] = (callforwardbusy[partner[1]]) & 0xff;

                currentState = GlobalClass_User_1Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:5) | calling -> oalert | [partner[1] != 1 and partner[1] != 4 and chan[partner[1]] = 255; record[partner[1]] := 1; chan[partner[1]] := (1 + 0 * 20); chan[1] := (partner[1] + 0 * 20)].
            private boolean execute_transition_calling_5() {
                // SLCO composite | [partner[1] != 1 and partner[1] != 4 and chan[partner[1]] = 255; record[partner[1]] := 1; chan[partner[1]] := ((1) + (0) * 20); chan[1] := ((partner[1]) + (0) * 20)] -> [partner[1] != 1 and partner[1] != 4 and chan[partner[1]] = 255; record[partner[1]] := 1; chan[partner[1]] := (1 + 0 * 20); chan[1] := (partner[1] + 0 * 20)].
                // SLCO expression | partner[1] != 1 and partner[1] != 4 and chan[partner[1]] = 255.
                if(!(partner[1] != 1 && partner[1] != 4 && chan[partner[1]] == 255)) {
                    return false;
                }
                // SLCO assignment | record[partner[1]] := 1.
                record[partner[1]] = (1) & 0xff;
                // SLCO assignment | chan[partner[1]] := (1 + 0 * 20).
                chan[partner[1]] = ((1 + 0 * 20)) & 0xff;
                // SLCO assignment | chan[1] := (partner[1] + 0 * 20).
                chan[1] = ((partner[1] + 0 * 20)) & 0xff;

                currentState = GlobalClass_User_1Thread.States.oalert;
                return true;
            }

            // SLCO transition (p:0, id:0) | busy -> idle | true | [true; chan[1] := 255; partner[1] := 255; dev := 1].
            private boolean execute_transition_busy_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [chan[1] := 255; partner[1] := 255; dev := 1] -> [true; chan[1] := 255; partner[1] := 255; dev := 1].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | chan[1] := 255.
                chan[1] = (255) & 0xff;
                // SLCO assignment | partner[1] := 255.
                partner[1] = (255) & 0xff;
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_1Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | oalert -> errorstate | (chan[1] % 20) != partner[1].
            private boolean execute_transition_oalert_0() {
                // SLCO expression | ((chan[1]) % 20) != partner[1] -> (chan[1] % 20) != partner[1].
                if(!((Math.floorMod(chan[1], 20)) != partner[1])) {
                    return false;
                }

                currentState = GlobalClass_User_1Thread.States.errorstate;
                return true;
            }

            // SLCO transition (p:0, id:1) | oalert -> oconnected | (chan[1] % 20) = partner[1] and (chan[1] / 20) = 1.
            private boolean execute_transition_oalert_1() {
                // SLCO expression | ((chan[1]) % 20) = partner[1] and ((chan[1]) / 20) = 1 -> (chan[1] % 20) = partner[1] and (chan[1] / 20) = 1.
                if(!((Math.floorMod(chan[1], 20)) == partner[1] && (chan[1] / 20) == 1)) {
                    return false;
                }

                currentState = GlobalClass_User_1Thread.States.oconnected;
                return true;
            }

            // SLCO transition (p:0, id:2) | oalert -> dveoringout | (chan[1] % 20) = partner[1] and (chan[1] / 20) = 0.
            private boolean execute_transition_oalert_2() {
                // SLCO expression | ((chan[1]) % 20) = partner[1] and ((chan[1]) / 20) = 0 -> (chan[1] % 20) = partner[1] and (chan[1] / 20) = 0.
                if(!((Math.floorMod(chan[1], 20)) == partner[1] && (chan[1] / 20) == 0)) {
                    return false;
                }

                currentState = GlobalClass_User_1Thread.States.dveoringout;
                return true;
            }

            // SLCO transition (p:0, id:0) | oconnected -> idle | true | [true; dev := 1; chan[1] := 255; chan[partner[1]] := 255].
            private boolean execute_transition_oconnected_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [dev := 1; chan[1] := 255; chan[partner[1]] := 255] -> [true; dev := 1; chan[1] := 255; chan[partner[1]] := 255].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | chan[1] := 255.
                chan[1] = (255) & 0xff;
                // SLCO assignment | chan[partner[1]] := 255.
                chan[partner[1]] = (255) & 0xff;

                currentState = GlobalClass_User_1Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | dveoringout -> idle | true | [true; dev := 1; chan[1] := 255; partner[1] := ((partner[1] % 20) + 0 * 20)].
            private boolean execute_transition_dveoringout_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [dev := 1; chan[1] := 255; partner[1] := ((((partner[1]) % 20)) + (0) * 20)] -> [true; dev := 1; chan[1] := 255; partner[1] := ((partner[1] % 20) + 0 * 20)].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | chan[1] := 255.
                chan[1] = (255) & 0xff;
                // SLCO assignment | partner[1] := ((partner[1] % 20) + 0 * 20).
                partner[1] = (((Math.floorMod(partner[1], 20)) + 0 * 20)) & 0xff;

                currentState = GlobalClass_User_1Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | unobtainable -> idle | true | [true; chan[1] := 255; partner[1] := 255; dev := 1].
            private boolean execute_transition_unobtainable_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [chan[1] := 255; partner[1] := 255; dev := 1] -> [true; chan[1] := 255; partner[1] := 255; dev := 1].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | chan[1] := 255.
                chan[1] = (255) & 0xff;
                // SLCO assignment | partner[1] := 255.
                partner[1] = (255) & 0xff;
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_1Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | ringback -> idle | true | [true; chan[1] := 255; partner[1] := 255; dev := 1].
            private boolean execute_transition_ringback_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [chan[1] := 255; partner[1] := 255; dev := 1] -> [true; chan[1] := 255; partner[1] := 255; dev := 1].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | chan[1] := 255.
                chan[1] = (255) & 0xff;
                // SLCO assignment | partner[1] := 255.
                partner[1] = (255) & 0xff;
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_1Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:1) | ringback -> calling | [record[1] != 255; partner[1] := record[1]].
            private boolean execute_transition_ringback_1() {
                // SLCO composite | [record[1] != 255; partner[1] := record[1]].
                // SLCO expression | record[1] != 255.
                if(!(record[1] != 255)) {
                    return false;
                }
                // SLCO assignment | partner[1] := record[1].
                partner[1] = (record[1]) & 0xff;

                currentState = GlobalClass_User_1Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:0) | talert -> errorstate | dev != 1 or chan[1] = 255.
            private boolean execute_transition_talert_0() {
                // SLCO expression | dev != 1 or chan[1] = 255.
                if(!(dev != 1 || chan[1] == 255)) {
                    return false;
                }

                currentState = GlobalClass_User_1Thread.States.errorstate;
                return true;
            }

            // SLCO transition (p:0, id:1) | talert -> tpickup | (chan[partner[1]] % 20) = 1.
            private boolean execute_transition_talert_1() {
                // SLCO expression | ((chan[partner[1]]) % 20) = 1 -> (chan[partner[1]] % 20) = 1.
                if(!((Math.floorMod(chan[partner[1]], 20)) == 1)) {
                    return false;
                }

                currentState = GlobalClass_User_1Thread.States.tpickup;
                return true;
            }

            // SLCO transition (p:0, id:2) | talert -> idle | (chan[partner[1]] % 20) != 1.
            private boolean execute_transition_talert_2() {
                // SLCO expression | ((chan[partner[1]]) % 20) != 1 -> (chan[partner[1]] % 20) != 1.
                if(!((Math.floorMod(chan[partner[1]], 20)) != 1)) {
                    return false;
                }

                currentState = GlobalClass_User_1Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | tpickup -> tconnected | [(chan[partner[1]] % 20) = 1 and (chan[partner[1]] / 20) = 0; dev := 0; chan[partner[1]] := (1 + 1 * 20); chan[1] := (partner[1] + 1 * 20)].
            private boolean execute_transition_tpickup_0() {
                // SLCO composite | [((chan[partner[1]]) % 20) = 1 and ((chan[partner[1]]) / 20) = 0; dev := 0; chan[partner[1]] := ((1) + (1) * 20); chan[1] := ((partner[1]) + (1) * 20)] -> [(chan[partner[1]] % 20) = 1 and (chan[partner[1]] / 20) = 0; dev := 0; chan[partner[1]] := (1 + 1 * 20); chan[1] := (partner[1] + 1 * 20)].
                // SLCO expression | (chan[partner[1]] % 20) = 1 and (chan[partner[1]] / 20) = 0.
                if(!((Math.floorMod(chan[partner[1]], 20)) == 1 && (chan[partner[1]] / 20) == 0)) {
                    return false;
                }
                // SLCO assignment | dev := 0.
                dev = (0) & 0xff;
                // SLCO assignment | chan[partner[1]] := (1 + 1 * 20).
                chan[partner[1]] = ((1 + 1 * 20)) & 0xff;
                // SLCO assignment | chan[1] := (partner[1] + 1 * 20).
                chan[1] = ((partner[1] + 1 * 20)) & 0xff;

                currentState = GlobalClass_User_1Thread.States.tconnected;
                return true;
            }

            // SLCO transition (p:0, id:1) | tpickup -> idle | [chan[partner[1]] = 255 or (chan[partner[1]] % 20) != 1; dev := 1; partner[1] := 255; chan[1] := 255].
            private boolean execute_transition_tpickup_1() {
                // SLCO composite | [chan[partner[1]] = 255 or ((chan[partner[1]]) % 20) != 1; dev := 1; partner[1] := 255; chan[1] := 255] -> [chan[partner[1]] = 255 or (chan[partner[1]] % 20) != 1; dev := 1; partner[1] := 255; chan[1] := 255].
                // SLCO expression | chan[partner[1]] = 255 or (chan[partner[1]] % 20) != 1.
                if(!(chan[partner[1]] == 255 || (Math.floorMod(chan[partner[1]], 20)) != 1)) {
                    return false;
                }
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | partner[1] := 255.
                partner[1] = (255) & 0xff;
                // SLCO assignment | chan[1] := 255.
                chan[1] = (255) & 0xff;

                currentState = GlobalClass_User_1Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | tconnected -> tconnected | [(chan[1] / 20) = 1 and dev = 0; dev := 1].
            private boolean execute_transition_tconnected_0() {
                // SLCO composite | [((chan[1]) / 20) = 1 and dev = 0; dev := 1] -> [(chan[1] / 20) = 1 and dev = 0; dev := 1].
                // SLCO expression | (chan[1] / 20) = 1 and dev = 0.
                if(!((chan[1] / 20) == 1 && dev == 0)) {
                    return false;
                }
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_1Thread.States.tconnected;
                return true;
            }

            // SLCO transition (p:0, id:1) | tconnected -> tconnected | [(chan[1] / 20) = 1 and dev = 1; dev := 0].
            private boolean execute_transition_tconnected_1() {
                // SLCO composite | [((chan[1]) / 20) = 1 and dev = 1; dev := 0] -> [(chan[1] / 20) = 1 and dev = 1; dev := 0].
                // SLCO expression | (chan[1] / 20) = 1 and dev = 1.
                if(!((chan[1] / 20) == 1 && dev == 1)) {
                    return false;
                }
                // SLCO assignment | dev := 0.
                dev = (0) & 0xff;

                currentState = GlobalClass_User_1Thread.States.tconnected;
                return true;
            }

            // SLCO transition (p:0, id:2) | tconnected -> idle | [(chan[1] / 20) = 0; partner[1] := 255; chan[1] := 255].
            private boolean execute_transition_tconnected_2() {
                // SLCO composite | [((chan[1]) / 20) = 0; partner[1] := 255; chan[1] := 255] -> [(chan[1] / 20) = 0; partner[1] := 255; chan[1] := 255].
                // SLCO expression | (chan[1] / 20) = 0.
                if(!((chan[1] / 20) == 0)) {
                    return false;
                }
                // SLCO assignment | partner[1] := 255.
                partner[1] = (255) & 0xff;
                // SLCO assignment | chan[1] := 255.
                chan[1] = (255) & 0xff;

                currentState = GlobalClass_User_1Thread.States.idle;
                return true;
            }

            // Attempt to fire a transition starting in state idle.
            private void exec_idle() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | idle -> dialing | [chan[1] = 255; dev := 0; chan[1] := (1 + 0 * 20)].
                if(execute_transition_idle_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | idle -> qi | [chan[1] != 255; partner[1] := (chan[1] % 20)].
                if(execute_transition_idle_1()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state dialing.
            private void exec_dialing() {
                // [N_DET.START]
                switch(random.nextInt(6)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | dialing -> idle | true | [true; dev := 1; chan[1] := 255].
                        if(execute_transition_dialing_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (p:0, id:1) | dialing -> calling | true | partner[1] := 0.
                        if(execute_transition_dialing_1()) {
                            return;
                        }
                    }
                    case 2 -> {
                        // SLCO transition (p:0, id:2) | dialing -> calling | true | partner[1] := 1.
                        if(execute_transition_dialing_2()) {
                            return;
                        }
                    }
                    case 3 -> {
                        // SLCO transition (p:0, id:3) | dialing -> calling | true | partner[1] := 2.
                        if(execute_transition_dialing_3()) {
                            return;
                        }
                    }
                    case 4 -> {
                        // SLCO transition (p:0, id:4) | dialing -> calling | true | partner[1] := 3.
                        if(execute_transition_dialing_4()) {
                            return;
                        }
                    }
                    case 5 -> {
                        // SLCO transition (p:0, id:5) | dialing -> calling | true | partner[1] := 4.
                        if(execute_transition_dialing_5()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state calling.
            private void exec_calling() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | calling -> busy | partner[1] = 1.
                if(execute_transition_calling_0()) {
                    return;
                }
                // SLCO expression | partner[1] = 4.
                if(partner[1] == 4) {
                    // [N_DET.START]
                    switch(random.nextInt(2)) {
                        case 0 -> {
                            // SLCO transition (p:0, id:1) | calling -> unobtainable | partner[1] = 4.
                            if(execute_transition_calling_1()) {
                                return;
                            }
                        }
                        case 1 -> {
                            // SLCO transition (p:0, id:2) | calling -> ringback | partner[1] = 4.
                            if(execute_transition_calling_2()) {
                                return;
                            }
                        }
                    }
                    // [N_DET.END]
                }
                // SLCO transition (p:0, id:3) | calling -> busy | [partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255 and callforwardbusy[partner[1]] = 255; record[partner[1]] := 1].
                if(execute_transition_calling_3()) {
                    return;
                }
                // SLCO transition (p:0, id:4) | calling -> calling | [partner[1] != 1 and partner[1] != 4 and chan[partner[1]] != 255 and callforwardbusy[partner[1]] != 255; record[partner[1]] := 1; partner[1] := callforwardbusy[partner[1]]].
                if(execute_transition_calling_4()) {
                    return;
                }
                // SLCO transition (p:0, id:5) | calling -> oalert | [partner[1] != 1 and partner[1] != 4 and chan[partner[1]] = 255; record[partner[1]] := 1; chan[partner[1]] := (1 + 0 * 20); chan[1] := (partner[1] + 0 * 20)].
                if(execute_transition_calling_5()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state busy.
            private void exec_busy() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | busy -> idle | true | [true; chan[1] := 255; partner[1] := 255; dev := 1].
                if(execute_transition_busy_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state qi.
            private void exec_qi() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | qi -> talert | (chan[partner[1]] % 20) = 1.
                if(execute_transition_qi_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | qi -> idle | [(chan[partner[1]] % 20) != 1; partner[1] := 255].
                if(execute_transition_qi_1()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state talert.
            private void exec_talert() {
                // [N_DET.START]
                switch(random.nextInt(2)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | talert -> errorstate | dev != 1 or chan[1] = 255.
                        if(execute_transition_talert_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // [DET.START]
                        // SLCO transition (p:0, id:1) | talert -> tpickup | (chan[partner[1]] % 20) = 1.
                        if(execute_transition_talert_1()) {
                            return;
                        }
                        // SLCO transition (p:0, id:2) | talert -> idle | (chan[partner[1]] % 20) != 1.
                        if(execute_transition_talert_2()) {
                            return;
                        }
                        // [DET.END]
                    }
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state unobtainable.
            private void exec_unobtainable() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | unobtainable -> idle | true | [true; chan[1] := 255; partner[1] := 255; dev := 1].
                if(execute_transition_unobtainable_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state oalert.
            private void exec_oalert() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | oalert -> errorstate | (chan[1] % 20) != partner[1].
                if(execute_transition_oalert_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | oalert -> oconnected | (chan[1] % 20) = partner[1] and (chan[1] / 20) = 1.
                if(execute_transition_oalert_1()) {
                    return;
                }
                // SLCO transition (p:0, id:2) | oalert -> dveoringout | (chan[1] % 20) = partner[1] and (chan[1] / 20) = 0.
                if(execute_transition_oalert_2()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state errorstate.
            private void exec_errorstate() {
                // There are no transitions starting in state errorstate.
            }

            // Attempt to fire a transition starting in state oconnected.
            private void exec_oconnected() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | oconnected -> idle | true | [true; dev := 1; chan[1] := 255; chan[partner[1]] := 255].
                if(execute_transition_oconnected_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state dveoringout.
            private void exec_dveoringout() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | dveoringout -> idle | true | [true; dev := 1; chan[1] := 255; partner[1] := ((partner[1] % 20) + 0 * 20)].
                if(execute_transition_dveoringout_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state tpickup.
            private void exec_tpickup() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | tpickup -> tconnected | [(chan[partner[1]] % 20) = 1 and (chan[partner[1]] / 20) = 0; dev := 0; chan[partner[1]] := (1 + 1 * 20); chan[1] := (partner[1] + 1 * 20)].
                if(execute_transition_tpickup_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | tpickup -> idle | [chan[partner[1]] = 255 or (chan[partner[1]] % 20) != 1; dev := 1; partner[1] := 255; chan[1] := 255].
                if(execute_transition_tpickup_1()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state tconnected.
            private void exec_tconnected() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | tconnected -> tconnected | [(chan[1] / 20) = 1 and dev = 0; dev := 1].
                if(execute_transition_tconnected_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | tconnected -> tconnected | [(chan[1] / 20) = 1 and dev = 1; dev := 0].
                if(execute_transition_tconnected_1()) {
                    return;
                }
                // SLCO transition (p:0, id:2) | tconnected -> idle | [(chan[1] / 20) = 0; partner[1] := 255; chan[1] := 255].
                if(execute_transition_tconnected_2()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state ringback.
            private void exec_ringback() {
                // [N_DET.START]
                switch(random.nextInt(2)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | ringback -> idle | true | [true; chan[1] := 255; partner[1] := 255; dev := 1].
                        if(execute_transition_ringback_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (p:0, id:1) | ringback -> calling | [record[1] != 255; partner[1] := record[1]].
                        if(execute_transition_ringback_1()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 30) {
                    switch(currentState) {
                        case idle -> exec_idle();
                        case dialing -> exec_dialing();
                        case calling -> exec_calling();
                        case busy -> exec_busy();
                        case qi -> exec_qi();
                        case talert -> exec_talert();
                        case unobtainable -> exec_unobtainable();
                        case oalert -> exec_oalert();
                        case errorstate -> exec_errorstate();
                        case oconnected -> exec_oconnected();
                        case dveoringout -> exec_dveoringout();
                        case tpickup -> exec_tpickup();
                        case tconnected -> exec_tconnected();
                        case ringback -> exec_ringback();
                    }
                }
            }

            // The thread's run method.
            public void run() {
                try {
                    exec();
                } catch(Exception e) {
                    lockManager.exception_unlock();
                    throw e;
                }
            }
        }

        // Define the states fot the state machine User_2.
        interface GlobalClass_User_2Thread_States {
            enum States {
                idle, 
                dialing, 
                calling, 
                busy, 
                qi, 
                talert, 
                unobtainable, 
                oalert, 
                errorstate, 
                oconnected, 
                dveoringout, 
                tpickup, 
                tconnected, 
                ringback
            }
        }

        // Representation of the SLCO state machine User_2.
        class GlobalClass_User_2Thread extends Thread implements GlobalClass_User_2Thread_States {
            // Current state
            private GlobalClass_User_2Thread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // Thread local variables.
            private int dev;
            private int mbit;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            GlobalClass_User_2Thread(LockManager lockManagerInstance) {
                currentState = GlobalClass_User_2Thread.States.idle;
                lockManager = lockManagerInstance;
                lock_ids = new int[0];
                target_locks = new int[0];
                random = new Random();

                // Variable instantiations.
                dev = (char) 1;
                mbit = (char) 0;
            }

            // SLCO transition (p:0, id:0) | idle -> dialing | [chan[2] = 255; dev := 0; chan[2] := (2 + 0 * 20)].
            private boolean execute_transition_idle_0() {
                // SLCO composite | [chan[2] = 255; dev := 0; chan[2] := ((2) + (0) * 20)] -> [chan[2] = 255; dev := 0; chan[2] := (2 + 0 * 20)].
                // SLCO expression | chan[2] = 255.
                if(!(chan[2] == 255)) {
                    return false;
                }
                // SLCO assignment | dev := 0.
                dev = (0) & 0xff;
                // SLCO assignment | chan[2] := (2 + 0 * 20).
                chan[2] = ((2 + 0 * 20)) & 0xff;

                currentState = GlobalClass_User_2Thread.States.dialing;
                return true;
            }

            // SLCO transition (p:0, id:1) | idle -> qi | [chan[2] != 255; partner[2] := (chan[2] % 20)].
            private boolean execute_transition_idle_1() {
                // SLCO composite | [chan[2] != 255; partner[2] := ((chan[2]) % 20)] -> [chan[2] != 255; partner[2] := (chan[2] % 20)].
                // SLCO expression | chan[2] != 255.
                if(!(chan[2] != 255)) {
                    return false;
                }
                // SLCO assignment | partner[2] := (chan[2] % 20).
                partner[2] = ((Math.floorMod(chan[2], 20))) & 0xff;

                currentState = GlobalClass_User_2Thread.States.qi;
                return true;
            }

            // SLCO transition (p:0, id:0) | qi -> talert | (chan[partner[2]] % 20) = 2.
            private boolean execute_transition_qi_0() {
                // SLCO expression | ((chan[partner[2]]) % 20) = 2 -> (chan[partner[2]] % 20) = 2.
                if(!((Math.floorMod(chan[partner[2]], 20)) == 2)) {
                    return false;
                }

                currentState = GlobalClass_User_2Thread.States.talert;
                return true;
            }

            // SLCO transition (p:0, id:1) | qi -> idle | [(chan[partner[2]] % 20) != 2; partner[2] := 255].
            private boolean execute_transition_qi_1() {
                // SLCO composite | [((chan[partner[2]]) % 20) != 2; partner[2] := 255] -> [(chan[partner[2]] % 20) != 2; partner[2] := 255].
                // SLCO expression | (chan[partner[2]] % 20) != 2.
                if(!((Math.floorMod(chan[partner[2]], 20)) != 2)) {
                    return false;
                }
                // SLCO assignment | partner[2] := 255.
                partner[2] = (255) & 0xff;

                currentState = GlobalClass_User_2Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | dialing -> idle | true | [true; dev := 1; chan[2] := 255].
            private boolean execute_transition_dialing_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [dev := 1; chan[2] := 255] -> [true; dev := 1; chan[2] := 255].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | chan[2] := 255.
                chan[2] = (255) & 0xff;

                currentState = GlobalClass_User_2Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:1) | dialing -> calling | true | partner[2] := 0.
            private boolean execute_transition_dialing_1() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[2] := 0] -> partner[2] := 0.
                partner[2] = (0) & 0xff;

                currentState = GlobalClass_User_2Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:2) | dialing -> calling | true | partner[2] := 1.
            private boolean execute_transition_dialing_2() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[2] := 1] -> partner[2] := 1.
                partner[2] = (1) & 0xff;

                currentState = GlobalClass_User_2Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:3) | dialing -> calling | true | partner[2] := 2.
            private boolean execute_transition_dialing_3() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[2] := 2] -> partner[2] := 2.
                partner[2] = (2) & 0xff;

                currentState = GlobalClass_User_2Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:4) | dialing -> calling | true | partner[2] := 3.
            private boolean execute_transition_dialing_4() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[2] := 3] -> partner[2] := 3.
                partner[2] = (3) & 0xff;

                currentState = GlobalClass_User_2Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:5) | dialing -> calling | true | partner[2] := 4.
            private boolean execute_transition_dialing_5() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[2] := 4] -> partner[2] := 4.
                partner[2] = (4) & 0xff;

                currentState = GlobalClass_User_2Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:0) | calling -> busy | partner[2] = 2.
            private boolean execute_transition_calling_0() {
                // SLCO expression | partner[2] = 2.
                if(!(partner[2] == 2)) {
                    return false;
                }

                currentState = GlobalClass_User_2Thread.States.busy;
                return true;
            }

            // SLCO transition (p:0, id:1) | calling -> unobtainable | partner[2] = 4.
            private boolean execute_transition_calling_1() {
                // SLCO expression | partner[2] = 4.
                if(!(partner[2] == 4)) {
                    return false;
                }

                currentState = GlobalClass_User_2Thread.States.unobtainable;
                return true;
            }

            // SLCO transition (p:0, id:2) | calling -> ringback | partner[2] = 4.
            private boolean execute_transition_calling_2() {
                // SLCO expression | partner[2] = 4.
                if(!(partner[2] == 4)) {
                    return false;
                }

                currentState = GlobalClass_User_2Thread.States.ringback;
                return true;
            }

            // SLCO transition (p:0, id:3) | calling -> busy | [partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255 and callforwardbusy[partner[2]] = 255; record[partner[2]] := 2].
            private boolean execute_transition_calling_3() {
                // SLCO composite | [partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255 and callforwardbusy[partner[2]] = 255; record[partner[2]] := 2].
                // SLCO expression | partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255 and callforwardbusy[partner[2]] = 255.
                if(!(partner[2] != 2 && partner[2] != 4 && chan[partner[2]] != 255 && callforwardbusy[partner[2]] == 255)) {
                    return false;
                }
                // SLCO assignment | record[partner[2]] := 2.
                record[partner[2]] = (2) & 0xff;

                currentState = GlobalClass_User_2Thread.States.busy;
                return true;
            }

            // SLCO transition (p:0, id:4) | calling -> calling | [partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255 and callforwardbusy[partner[2]] != 255; record[partner[2]] := 2; partner[2] := callforwardbusy[partner[2]]].
            private boolean execute_transition_calling_4() {
                // SLCO composite | [partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255 and callforwardbusy[partner[2]] != 255; record[partner[2]] := 2; partner[2] := callforwardbusy[partner[2]]].
                // SLCO expression | partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255 and callforwardbusy[partner[2]] != 255.
                if(!(partner[2] != 2 && partner[2] != 4 && chan[partner[2]] != 255 && callforwardbusy[partner[2]] != 255)) {
                    return false;
                }
                // SLCO assignment | record[partner[2]] := 2.
                record[partner[2]] = (2) & 0xff;
                // SLCO assignment | partner[2] := callforwardbusy[partner[2]].
                partner[2] = (callforwardbusy[partner[2]]) & 0xff;

                currentState = GlobalClass_User_2Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:5) | calling -> oalert | [partner[2] != 2 and partner[2] != 4 and chan[partner[2]] = 255; record[partner[2]] := 2; chan[partner[2]] := (2 + 0 * 20); chan[2] := (partner[2] + 0 * 20)].
            private boolean execute_transition_calling_5() {
                // SLCO composite | [partner[2] != 2 and partner[2] != 4 and chan[partner[2]] = 255; record[partner[2]] := 2; chan[partner[2]] := ((2) + (0) * 20); chan[2] := ((partner[2]) + (0) * 20)] -> [partner[2] != 2 and partner[2] != 4 and chan[partner[2]] = 255; record[partner[2]] := 2; chan[partner[2]] := (2 + 0 * 20); chan[2] := (partner[2] + 0 * 20)].
                // SLCO expression | partner[2] != 2 and partner[2] != 4 and chan[partner[2]] = 255.
                if(!(partner[2] != 2 && partner[2] != 4 && chan[partner[2]] == 255)) {
                    return false;
                }
                // SLCO assignment | record[partner[2]] := 2.
                record[partner[2]] = (2) & 0xff;
                // SLCO assignment | chan[partner[2]] := (2 + 0 * 20).
                chan[partner[2]] = ((2 + 0 * 20)) & 0xff;
                // SLCO assignment | chan[2] := (partner[2] + 0 * 20).
                chan[2] = ((partner[2] + 0 * 20)) & 0xff;

                currentState = GlobalClass_User_2Thread.States.oalert;
                return true;
            }

            // SLCO transition (p:0, id:0) | busy -> idle | true | [true; chan[2] := 255; partner[2] := 255; dev := 1].
            private boolean execute_transition_busy_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [chan[2] := 255; partner[2] := 255; dev := 1] -> [true; chan[2] := 255; partner[2] := 255; dev := 1].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | chan[2] := 255.
                chan[2] = (255) & 0xff;
                // SLCO assignment | partner[2] := 255.
                partner[2] = (255) & 0xff;
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_2Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | oalert -> errorstate | (chan[2] % 20) != partner[2].
            private boolean execute_transition_oalert_0() {
                // SLCO expression | ((chan[2]) % 20) != partner[2] -> (chan[2] % 20) != partner[2].
                if(!((Math.floorMod(chan[2], 20)) != partner[2])) {
                    return false;
                }

                currentState = GlobalClass_User_2Thread.States.errorstate;
                return true;
            }

            // SLCO transition (p:0, id:1) | oalert -> oconnected | (chan[2] % 20) = partner[2] and (chan[2] / 20) = 1.
            private boolean execute_transition_oalert_1() {
                // SLCO expression | ((chan[2]) % 20) = partner[2] and ((chan[2]) / 20) = 1 -> (chan[2] % 20) = partner[2] and (chan[2] / 20) = 1.
                if(!((Math.floorMod(chan[2], 20)) == partner[2] && (chan[2] / 20) == 1)) {
                    return false;
                }

                currentState = GlobalClass_User_2Thread.States.oconnected;
                return true;
            }

            // SLCO transition (p:0, id:2) | oalert -> dveoringout | (chan[2] % 20) = partner[2] and (chan[2] / 20) = 0.
            private boolean execute_transition_oalert_2() {
                // SLCO expression | ((chan[2]) % 20) = partner[2] and ((chan[2]) / 20) = 0 -> (chan[2] % 20) = partner[2] and (chan[2] / 20) = 0.
                if(!((Math.floorMod(chan[2], 20)) == partner[2] && (chan[2] / 20) == 0)) {
                    return false;
                }

                currentState = GlobalClass_User_2Thread.States.dveoringout;
                return true;
            }

            // SLCO transition (p:0, id:0) | oconnected -> idle | true | [true; dev := 1; chan[2] := 255; chan[partner[2]] := 255].
            private boolean execute_transition_oconnected_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [dev := 1; chan[2] := 255; chan[partner[2]] := 255] -> [true; dev := 1; chan[2] := 255; chan[partner[2]] := 255].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | chan[2] := 255.
                chan[2] = (255) & 0xff;
                // SLCO assignment | chan[partner[2]] := 255.
                chan[partner[2]] = (255) & 0xff;

                currentState = GlobalClass_User_2Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | dveoringout -> idle | true | [true; dev := 1; chan[2] := 255; partner[2] := ((partner[2] % 20) + 0 * 20)].
            private boolean execute_transition_dveoringout_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [dev := 1; chan[2] := 255; partner[2] := ((((partner[2]) % 20)) + (0) * 20)] -> [true; dev := 1; chan[2] := 255; partner[2] := ((partner[2] % 20) + 0 * 20)].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | chan[2] := 255.
                chan[2] = (255) & 0xff;
                // SLCO assignment | partner[2] := ((partner[2] % 20) + 0 * 20).
                partner[2] = (((Math.floorMod(partner[2], 20)) + 0 * 20)) & 0xff;

                currentState = GlobalClass_User_2Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | unobtainable -> idle | true | [true; chan[2] := 255; partner[2] := 255; dev := 1].
            private boolean execute_transition_unobtainable_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [chan[2] := 255; partner[2] := 255; dev := 1] -> [true; chan[2] := 255; partner[2] := 255; dev := 1].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | chan[2] := 255.
                chan[2] = (255) & 0xff;
                // SLCO assignment | partner[2] := 255.
                partner[2] = (255) & 0xff;
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_2Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | ringback -> idle | true | [true; chan[2] := 255; partner[2] := 255; dev := 1].
            private boolean execute_transition_ringback_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [chan[2] := 255; partner[2] := 255; dev := 1] -> [true; chan[2] := 255; partner[2] := 255; dev := 1].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | chan[2] := 255.
                chan[2] = (255) & 0xff;
                // SLCO assignment | partner[2] := 255.
                partner[2] = (255) & 0xff;
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_2Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:1) | ringback -> calling | [record[2] != 255; partner[2] := record[2]].
            private boolean execute_transition_ringback_1() {
                // SLCO composite | [record[2] != 255; partner[2] := record[2]].
                // SLCO expression | record[2] != 255.
                if(!(record[2] != 255)) {
                    return false;
                }
                // SLCO assignment | partner[2] := record[2].
                partner[2] = (record[2]) & 0xff;

                currentState = GlobalClass_User_2Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:0) | talert -> errorstate | dev != 1 or chan[2] = 255.
            private boolean execute_transition_talert_0() {
                // SLCO expression | dev != 1 or chan[2] = 255.
                if(!(dev != 1 || chan[2] == 255)) {
                    return false;
                }

                currentState = GlobalClass_User_2Thread.States.errorstate;
                return true;
            }

            // SLCO transition (p:0, id:1) | talert -> tpickup | (chan[partner[2]] % 20) = 2.
            private boolean execute_transition_talert_1() {
                // SLCO expression | ((chan[partner[2]]) % 20) = 2 -> (chan[partner[2]] % 20) = 2.
                if(!((Math.floorMod(chan[partner[2]], 20)) == 2)) {
                    return false;
                }

                currentState = GlobalClass_User_2Thread.States.tpickup;
                return true;
            }

            // SLCO transition (p:0, id:2) | talert -> idle | (chan[partner[2]] % 20) != 2.
            private boolean execute_transition_talert_2() {
                // SLCO expression | ((chan[partner[2]]) % 20) != 2 -> (chan[partner[2]] % 20) != 2.
                if(!((Math.floorMod(chan[partner[2]], 20)) != 2)) {
                    return false;
                }

                currentState = GlobalClass_User_2Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | tpickup -> tconnected | [(chan[partner[2]] % 20) = 2 and (chan[partner[2]] / 20) = 0; dev := 0; chan[partner[2]] := (2 + 1 * 20); chan[2] := (partner[2] + 1 * 20)].
            private boolean execute_transition_tpickup_0() {
                // SLCO composite | [((chan[partner[2]]) % 20) = 2 and ((chan[partner[2]]) / 20) = 0; dev := 0; chan[partner[2]] := ((2) + (1) * 20); chan[2] := ((partner[2]) + (1) * 20)] -> [(chan[partner[2]] % 20) = 2 and (chan[partner[2]] / 20) = 0; dev := 0; chan[partner[2]] := (2 + 1 * 20); chan[2] := (partner[2] + 1 * 20)].
                // SLCO expression | (chan[partner[2]] % 20) = 2 and (chan[partner[2]] / 20) = 0.
                if(!((Math.floorMod(chan[partner[2]], 20)) == 2 && (chan[partner[2]] / 20) == 0)) {
                    return false;
                }
                // SLCO assignment | dev := 0.
                dev = (0) & 0xff;
                // SLCO assignment | chan[partner[2]] := (2 + 1 * 20).
                chan[partner[2]] = ((2 + 1 * 20)) & 0xff;
                // SLCO assignment | chan[2] := (partner[2] + 1 * 20).
                chan[2] = ((partner[2] + 1 * 20)) & 0xff;

                currentState = GlobalClass_User_2Thread.States.tconnected;
                return true;
            }

            // SLCO transition (p:0, id:1) | tpickup -> idle | [chan[partner[2]] = 255 or (chan[partner[2]] % 20) != 2; dev := 1; partner[2] := 255; chan[2] := 255].
            private boolean execute_transition_tpickup_1() {
                // SLCO composite | [chan[partner[2]] = 255 or ((chan[partner[2]]) % 20) != 2; dev := 1; partner[2] := 255; chan[2] := 255] -> [chan[partner[2]] = 255 or (chan[partner[2]] % 20) != 2; dev := 1; partner[2] := 255; chan[2] := 255].
                // SLCO expression | chan[partner[2]] = 255 or (chan[partner[2]] % 20) != 2.
                if(!(chan[partner[2]] == 255 || (Math.floorMod(chan[partner[2]], 20)) != 2)) {
                    return false;
                }
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | partner[2] := 255.
                partner[2] = (255) & 0xff;
                // SLCO assignment | chan[2] := 255.
                chan[2] = (255) & 0xff;

                currentState = GlobalClass_User_2Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | tconnected -> tconnected | [(chan[2] / 20) = 1 and dev = 0; dev := 1].
            private boolean execute_transition_tconnected_0() {
                // SLCO composite | [((chan[2]) / 20) = 1 and dev = 0; dev := 1] -> [(chan[2] / 20) = 1 and dev = 0; dev := 1].
                // SLCO expression | (chan[2] / 20) = 1 and dev = 0.
                if(!((chan[2] / 20) == 1 && dev == 0)) {
                    return false;
                }
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_2Thread.States.tconnected;
                return true;
            }

            // SLCO transition (p:0, id:1) | tconnected -> tconnected | [(chan[2] / 20) = 1 and dev = 1; dev := 0].
            private boolean execute_transition_tconnected_1() {
                // SLCO composite | [((chan[2]) / 20) = 1 and dev = 1; dev := 0] -> [(chan[2] / 20) = 1 and dev = 1; dev := 0].
                // SLCO expression | (chan[2] / 20) = 1 and dev = 1.
                if(!((chan[2] / 20) == 1 && dev == 1)) {
                    return false;
                }
                // SLCO assignment | dev := 0.
                dev = (0) & 0xff;

                currentState = GlobalClass_User_2Thread.States.tconnected;
                return true;
            }

            // SLCO transition (p:0, id:2) | tconnected -> idle | [(chan[2] / 20) = 0; partner[2] := 255; chan[2] := 255].
            private boolean execute_transition_tconnected_2() {
                // SLCO composite | [((chan[2]) / 20) = 0; partner[2] := 255; chan[2] := 255] -> [(chan[2] / 20) = 0; partner[2] := 255; chan[2] := 255].
                // SLCO expression | (chan[2] / 20) = 0.
                if(!((chan[2] / 20) == 0)) {
                    return false;
                }
                // SLCO assignment | partner[2] := 255.
                partner[2] = (255) & 0xff;
                // SLCO assignment | chan[2] := 255.
                chan[2] = (255) & 0xff;

                currentState = GlobalClass_User_2Thread.States.idle;
                return true;
            }

            // Attempt to fire a transition starting in state idle.
            private void exec_idle() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | idle -> dialing | [chan[2] = 255; dev := 0; chan[2] := (2 + 0 * 20)].
                if(execute_transition_idle_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | idle -> qi | [chan[2] != 255; partner[2] := (chan[2] % 20)].
                if(execute_transition_idle_1()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state dialing.
            private void exec_dialing() {
                // [N_DET.START]
                switch(random.nextInt(6)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | dialing -> idle | true | [true; dev := 1; chan[2] := 255].
                        if(execute_transition_dialing_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (p:0, id:1) | dialing -> calling | true | partner[2] := 0.
                        if(execute_transition_dialing_1()) {
                            return;
                        }
                    }
                    case 2 -> {
                        // SLCO transition (p:0, id:2) | dialing -> calling | true | partner[2] := 1.
                        if(execute_transition_dialing_2()) {
                            return;
                        }
                    }
                    case 3 -> {
                        // SLCO transition (p:0, id:3) | dialing -> calling | true | partner[2] := 2.
                        if(execute_transition_dialing_3()) {
                            return;
                        }
                    }
                    case 4 -> {
                        // SLCO transition (p:0, id:4) | dialing -> calling | true | partner[2] := 3.
                        if(execute_transition_dialing_4()) {
                            return;
                        }
                    }
                    case 5 -> {
                        // SLCO transition (p:0, id:5) | dialing -> calling | true | partner[2] := 4.
                        if(execute_transition_dialing_5()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state calling.
            private void exec_calling() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | calling -> busy | partner[2] = 2.
                if(execute_transition_calling_0()) {
                    return;
                }
                // SLCO expression | partner[2] = 4.
                if(partner[2] == 4) {
                    // [N_DET.START]
                    switch(random.nextInt(2)) {
                        case 0 -> {
                            // SLCO transition (p:0, id:1) | calling -> unobtainable | partner[2] = 4.
                            if(execute_transition_calling_1()) {
                                return;
                            }
                        }
                        case 1 -> {
                            // SLCO transition (p:0, id:2) | calling -> ringback | partner[2] = 4.
                            if(execute_transition_calling_2()) {
                                return;
                            }
                        }
                    }
                    // [N_DET.END]
                }
                // SLCO transition (p:0, id:3) | calling -> busy | [partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255 and callforwardbusy[partner[2]] = 255; record[partner[2]] := 2].
                if(execute_transition_calling_3()) {
                    return;
                }
                // SLCO transition (p:0, id:4) | calling -> calling | [partner[2] != 2 and partner[2] != 4 and chan[partner[2]] != 255 and callforwardbusy[partner[2]] != 255; record[partner[2]] := 2; partner[2] := callforwardbusy[partner[2]]].
                if(execute_transition_calling_4()) {
                    return;
                }
                // SLCO transition (p:0, id:5) | calling -> oalert | [partner[2] != 2 and partner[2] != 4 and chan[partner[2]] = 255; record[partner[2]] := 2; chan[partner[2]] := (2 + 0 * 20); chan[2] := (partner[2] + 0 * 20)].
                if(execute_transition_calling_5()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state busy.
            private void exec_busy() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | busy -> idle | true | [true; chan[2] := 255; partner[2] := 255; dev := 1].
                if(execute_transition_busy_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state qi.
            private void exec_qi() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | qi -> talert | (chan[partner[2]] % 20) = 2.
                if(execute_transition_qi_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | qi -> idle | [(chan[partner[2]] % 20) != 2; partner[2] := 255].
                if(execute_transition_qi_1()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state talert.
            private void exec_talert() {
                // [N_DET.START]
                switch(random.nextInt(2)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | talert -> errorstate | dev != 1 or chan[2] = 255.
                        if(execute_transition_talert_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // [DET.START]
                        // SLCO transition (p:0, id:1) | talert -> tpickup | (chan[partner[2]] % 20) = 2.
                        if(execute_transition_talert_1()) {
                            return;
                        }
                        // SLCO transition (p:0, id:2) | talert -> idle | (chan[partner[2]] % 20) != 2.
                        if(execute_transition_talert_2()) {
                            return;
                        }
                        // [DET.END]
                    }
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state unobtainable.
            private void exec_unobtainable() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | unobtainable -> idle | true | [true; chan[2] := 255; partner[2] := 255; dev := 1].
                if(execute_transition_unobtainable_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state oalert.
            private void exec_oalert() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | oalert -> errorstate | (chan[2] % 20) != partner[2].
                if(execute_transition_oalert_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | oalert -> oconnected | (chan[2] % 20) = partner[2] and (chan[2] / 20) = 1.
                if(execute_transition_oalert_1()) {
                    return;
                }
                // SLCO transition (p:0, id:2) | oalert -> dveoringout | (chan[2] % 20) = partner[2] and (chan[2] / 20) = 0.
                if(execute_transition_oalert_2()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state errorstate.
            private void exec_errorstate() {
                // There are no transitions starting in state errorstate.
            }

            // Attempt to fire a transition starting in state oconnected.
            private void exec_oconnected() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | oconnected -> idle | true | [true; dev := 1; chan[2] := 255; chan[partner[2]] := 255].
                if(execute_transition_oconnected_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state dveoringout.
            private void exec_dveoringout() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | dveoringout -> idle | true | [true; dev := 1; chan[2] := 255; partner[2] := ((partner[2] % 20) + 0 * 20)].
                if(execute_transition_dveoringout_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state tpickup.
            private void exec_tpickup() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | tpickup -> tconnected | [(chan[partner[2]] % 20) = 2 and (chan[partner[2]] / 20) = 0; dev := 0; chan[partner[2]] := (2 + 1 * 20); chan[2] := (partner[2] + 1 * 20)].
                if(execute_transition_tpickup_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | tpickup -> idle | [chan[partner[2]] = 255 or (chan[partner[2]] % 20) != 2; dev := 1; partner[2] := 255; chan[2] := 255].
                if(execute_transition_tpickup_1()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state tconnected.
            private void exec_tconnected() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | tconnected -> tconnected | [(chan[2] / 20) = 1 and dev = 0; dev := 1].
                if(execute_transition_tconnected_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | tconnected -> tconnected | [(chan[2] / 20) = 1 and dev = 1; dev := 0].
                if(execute_transition_tconnected_1()) {
                    return;
                }
                // SLCO transition (p:0, id:2) | tconnected -> idle | [(chan[2] / 20) = 0; partner[2] := 255; chan[2] := 255].
                if(execute_transition_tconnected_2()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state ringback.
            private void exec_ringback() {
                // [N_DET.START]
                switch(random.nextInt(2)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | ringback -> idle | true | [true; chan[2] := 255; partner[2] := 255; dev := 1].
                        if(execute_transition_ringback_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (p:0, id:1) | ringback -> calling | [record[2] != 255; partner[2] := record[2]].
                        if(execute_transition_ringback_1()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 30) {
                    switch(currentState) {
                        case idle -> exec_idle();
                        case dialing -> exec_dialing();
                        case calling -> exec_calling();
                        case busy -> exec_busy();
                        case qi -> exec_qi();
                        case talert -> exec_talert();
                        case unobtainable -> exec_unobtainable();
                        case oalert -> exec_oalert();
                        case errorstate -> exec_errorstate();
                        case oconnected -> exec_oconnected();
                        case dveoringout -> exec_dveoringout();
                        case tpickup -> exec_tpickup();
                        case tconnected -> exec_tconnected();
                        case ringback -> exec_ringback();
                    }
                }
            }

            // The thread's run method.
            public void run() {
                try {
                    exec();
                } catch(Exception e) {
                    lockManager.exception_unlock();
                    throw e;
                }
            }
        }

        // Define the states fot the state machine User_3.
        interface GlobalClass_User_3Thread_States {
            enum States {
                idle, 
                dialing, 
                calling, 
                busy, 
                qi, 
                talert, 
                unobtainable, 
                oalert, 
                errorstate, 
                oconnected, 
                dveoringout, 
                tpickup, 
                tconnected, 
                ringback
            }
        }

        // Representation of the SLCO state machine User_3.
        class GlobalClass_User_3Thread extends Thread implements GlobalClass_User_3Thread_States {
            // Current state
            private GlobalClass_User_3Thread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // Thread local variables.
            private int dev;
            private int mbit;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            GlobalClass_User_3Thread(LockManager lockManagerInstance) {
                currentState = GlobalClass_User_3Thread.States.idle;
                lockManager = lockManagerInstance;
                lock_ids = new int[0];
                target_locks = new int[0];
                random = new Random();

                // Variable instantiations.
                dev = (char) 1;
                mbit = (char) 0;
            }

            // SLCO transition (p:0, id:0) | idle -> dialing | [chan[3] = 255; dev := 0; chan[3] := (3 + 0 * 20)].
            private boolean execute_transition_idle_0() {
                // SLCO composite | [chan[3] = 255; dev := 0; chan[3] := ((3) + (0) * 20)] -> [chan[3] = 255; dev := 0; chan[3] := (3 + 0 * 20)].
                // SLCO expression | chan[3] = 255.
                if(!(chan[3] == 255)) {
                    return false;
                }
                // SLCO assignment | dev := 0.
                dev = (0) & 0xff;
                // SLCO assignment | chan[3] := (3 + 0 * 20).
                chan[3] = ((3 + 0 * 20)) & 0xff;

                currentState = GlobalClass_User_3Thread.States.dialing;
                return true;
            }

            // SLCO transition (p:0, id:1) | idle -> qi | [chan[3] != 255; partner[3] := (chan[3] % 20)].
            private boolean execute_transition_idle_1() {
                // SLCO composite | [chan[3] != 255; partner[3] := ((chan[3]) % 20)] -> [chan[3] != 255; partner[3] := (chan[3] % 20)].
                // SLCO expression | chan[3] != 255.
                if(!(chan[3] != 255)) {
                    return false;
                }
                // SLCO assignment | partner[3] := (chan[3] % 20).
                partner[3] = ((Math.floorMod(chan[3], 20))) & 0xff;

                currentState = GlobalClass_User_3Thread.States.qi;
                return true;
            }

            // SLCO transition (p:0, id:0) | qi -> talert | (chan[partner[3]] % 20) = 3.
            private boolean execute_transition_qi_0() {
                // SLCO expression | ((chan[partner[3]]) % 20) = 3 -> (chan[partner[3]] % 20) = 3.
                if(!((Math.floorMod(chan[partner[3]], 20)) == 3)) {
                    return false;
                }

                currentState = GlobalClass_User_3Thread.States.talert;
                return true;
            }

            // SLCO transition (p:0, id:1) | qi -> idle | [(chan[partner[3]] % 20) != 3; partner[3] := 255].
            private boolean execute_transition_qi_1() {
                // SLCO composite | [((chan[partner[3]]) % 20) != 3; partner[3] := 255] -> [(chan[partner[3]] % 20) != 3; partner[3] := 255].
                // SLCO expression | (chan[partner[3]] % 20) != 3.
                if(!((Math.floorMod(chan[partner[3]], 20)) != 3)) {
                    return false;
                }
                // SLCO assignment | partner[3] := 255.
                partner[3] = (255) & 0xff;

                currentState = GlobalClass_User_3Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | dialing -> idle | true | [true; dev := 1; chan[3] := 255].
            private boolean execute_transition_dialing_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [dev := 1; chan[3] := 255] -> [true; dev := 1; chan[3] := 255].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | chan[3] := 255.
                chan[3] = (255) & 0xff;

                currentState = GlobalClass_User_3Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:1) | dialing -> calling | true | partner[3] := 0.
            private boolean execute_transition_dialing_1() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[3] := 0] -> partner[3] := 0.
                partner[3] = (0) & 0xff;

                currentState = GlobalClass_User_3Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:2) | dialing -> calling | true | partner[3] := 1.
            private boolean execute_transition_dialing_2() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[3] := 1] -> partner[3] := 1.
                partner[3] = (1) & 0xff;

                currentState = GlobalClass_User_3Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:3) | dialing -> calling | true | partner[3] := 2.
            private boolean execute_transition_dialing_3() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[3] := 2] -> partner[3] := 2.
                partner[3] = (2) & 0xff;

                currentState = GlobalClass_User_3Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:4) | dialing -> calling | true | partner[3] := 3.
            private boolean execute_transition_dialing_4() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[3] := 3] -> partner[3] := 3.
                partner[3] = (3) & 0xff;

                currentState = GlobalClass_User_3Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:5) | dialing -> calling | true | partner[3] := 4.
            private boolean execute_transition_dialing_5() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | [partner[3] := 4] -> partner[3] := 4.
                partner[3] = (4) & 0xff;

                currentState = GlobalClass_User_3Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:0) | calling -> busy | partner[3] = 3.
            private boolean execute_transition_calling_0() {
                // SLCO expression | partner[3] = 3.
                if(!(partner[3] == 3)) {
                    return false;
                }

                currentState = GlobalClass_User_3Thread.States.busy;
                return true;
            }

            // SLCO transition (p:0, id:1) | calling -> unobtainable | partner[3] = 4.
            private boolean execute_transition_calling_1() {
                // SLCO expression | partner[3] = 4.
                if(!(partner[3] == 4)) {
                    return false;
                }

                currentState = GlobalClass_User_3Thread.States.unobtainable;
                return true;
            }

            // SLCO transition (p:0, id:2) | calling -> ringback | partner[3] = 4.
            private boolean execute_transition_calling_2() {
                // SLCO expression | partner[3] = 4.
                if(!(partner[3] == 4)) {
                    return false;
                }

                currentState = GlobalClass_User_3Thread.States.ringback;
                return true;
            }

            // SLCO transition (p:0, id:3) | calling -> busy | [partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255 and callforwardbusy[partner[3]] = 255; record[partner[3]] := 3].
            private boolean execute_transition_calling_3() {
                // SLCO composite | [partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255 and callforwardbusy[partner[3]] = 255; record[partner[3]] := 3].
                // SLCO expression | partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255 and callforwardbusy[partner[3]] = 255.
                if(!(partner[3] != 3 && partner[3] != 4 && chan[partner[3]] != 255 && callforwardbusy[partner[3]] == 255)) {
                    return false;
                }
                // SLCO assignment | record[partner[3]] := 3.
                record[partner[3]] = (3) & 0xff;

                currentState = GlobalClass_User_3Thread.States.busy;
                return true;
            }

            // SLCO transition (p:0, id:4) | calling -> calling | [partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255 and callforwardbusy[partner[3]] != 255; record[partner[3]] := 3; partner[3] := callforwardbusy[partner[3]]].
            private boolean execute_transition_calling_4() {
                // SLCO composite | [partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255 and callforwardbusy[partner[3]] != 255; record[partner[3]] := 3; partner[3] := callforwardbusy[partner[3]]].
                // SLCO expression | partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255 and callforwardbusy[partner[3]] != 255.
                if(!(partner[3] != 3 && partner[3] != 4 && chan[partner[3]] != 255 && callforwardbusy[partner[3]] != 255)) {
                    return false;
                }
                // SLCO assignment | record[partner[3]] := 3.
                record[partner[3]] = (3) & 0xff;
                // SLCO assignment | partner[3] := callforwardbusy[partner[3]].
                partner[3] = (callforwardbusy[partner[3]]) & 0xff;

                currentState = GlobalClass_User_3Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:5) | calling -> oalert | [partner[3] != 3 and partner[3] != 4 and chan[partner[3]] = 255; record[partner[3]] := 3; chan[partner[3]] := (3 + 0 * 20); chan[3] := (partner[3] + 0 * 20)].
            private boolean execute_transition_calling_5() {
                // SLCO composite | [partner[3] != 3 and partner[3] != 4 and chan[partner[3]] = 255; record[partner[3]] := 3; chan[partner[3]] := ((3) + (0) * 20); chan[3] := ((partner[3]) + (0) * 20)] -> [partner[3] != 3 and partner[3] != 4 and chan[partner[3]] = 255; record[partner[3]] := 3; chan[partner[3]] := (3 + 0 * 20); chan[3] := (partner[3] + 0 * 20)].
                // SLCO expression | partner[3] != 3 and partner[3] != 4 and chan[partner[3]] = 255.
                if(!(partner[3] != 3 && partner[3] != 4 && chan[partner[3]] == 255)) {
                    return false;
                }
                // SLCO assignment | record[partner[3]] := 3.
                record[partner[3]] = (3) & 0xff;
                // SLCO assignment | chan[partner[3]] := (3 + 0 * 20).
                chan[partner[3]] = ((3 + 0 * 20)) & 0xff;
                // SLCO assignment | chan[3] := (partner[3] + 0 * 20).
                chan[3] = ((partner[3] + 0 * 20)) & 0xff;

                currentState = GlobalClass_User_3Thread.States.oalert;
                return true;
            }

            // SLCO transition (p:0, id:0) | busy -> idle | true | [true; chan[3] := 255; partner[3] := 255; dev := 1].
            private boolean execute_transition_busy_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [chan[3] := 255; partner[3] := 255; dev := 1] -> [true; chan[3] := 255; partner[3] := 255; dev := 1].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | chan[3] := 255.
                chan[3] = (255) & 0xff;
                // SLCO assignment | partner[3] := 255.
                partner[3] = (255) & 0xff;
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_3Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | oalert -> errorstate | (chan[3] % 20) != partner[3].
            private boolean execute_transition_oalert_0() {
                // SLCO expression | ((chan[3]) % 20) != partner[3] -> (chan[3] % 20) != partner[3].
                if(!((Math.floorMod(chan[3], 20)) != partner[3])) {
                    return false;
                }

                currentState = GlobalClass_User_3Thread.States.errorstate;
                return true;
            }

            // SLCO transition (p:0, id:1) | oalert -> oconnected | (chan[3] % 20) = partner[3] and (chan[3] / 20) = 1.
            private boolean execute_transition_oalert_1() {
                // SLCO expression | ((chan[3]) % 20) = partner[3] and ((chan[3]) / 20) = 1 -> (chan[3] % 20) = partner[3] and (chan[3] / 20) = 1.
                if(!((Math.floorMod(chan[3], 20)) == partner[3] && (chan[3] / 20) == 1)) {
                    return false;
                }

                currentState = GlobalClass_User_3Thread.States.oconnected;
                return true;
            }

            // SLCO transition (p:0, id:2) | oalert -> dveoringout | (chan[3] % 20) = partner[3] and (chan[3] / 20) = 0.
            private boolean execute_transition_oalert_2() {
                // SLCO expression | ((chan[3]) % 20) = partner[3] and ((chan[3]) / 20) = 0 -> (chan[3] % 20) = partner[3] and (chan[3] / 20) = 0.
                if(!((Math.floorMod(chan[3], 20)) == partner[3] && (chan[3] / 20) == 0)) {
                    return false;
                }

                currentState = GlobalClass_User_3Thread.States.dveoringout;
                return true;
            }

            // SLCO transition (p:0, id:0) | oconnected -> idle | true | [true; dev := 1; chan[3] := 255; chan[partner[3]] := 255].
            private boolean execute_transition_oconnected_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [dev := 1; chan[3] := 255; chan[partner[3]] := 255] -> [true; dev := 1; chan[3] := 255; chan[partner[3]] := 255].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | chan[3] := 255.
                chan[3] = (255) & 0xff;
                // SLCO assignment | chan[partner[3]] := 255.
                chan[partner[3]] = (255) & 0xff;

                currentState = GlobalClass_User_3Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | dveoringout -> idle | true | [true; dev := 1; chan[3] := 255; partner[3] := ((partner[3] % 20) + 0 * 20)].
            private boolean execute_transition_dveoringout_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [dev := 1; chan[3] := 255; partner[3] := ((((partner[3]) % 20)) + (0) * 20)] -> [true; dev := 1; chan[3] := 255; partner[3] := ((partner[3] % 20) + 0 * 20)].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | chan[3] := 255.
                chan[3] = (255) & 0xff;
                // SLCO assignment | partner[3] := ((partner[3] % 20) + 0 * 20).
                partner[3] = (((Math.floorMod(partner[3], 20)) + 0 * 20)) & 0xff;

                currentState = GlobalClass_User_3Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | unobtainable -> idle | true | [true; chan[3] := 255; partner[3] := 255; dev := 1].
            private boolean execute_transition_unobtainable_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [chan[3] := 255; partner[3] := 255; dev := 1] -> [true; chan[3] := 255; partner[3] := 255; dev := 1].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | chan[3] := 255.
                chan[3] = (255) & 0xff;
                // SLCO assignment | partner[3] := 255.
                partner[3] = (255) & 0xff;
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_3Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | ringback -> idle | true | [true; chan[3] := 255; partner[3] := 255; dev := 1].
            private boolean execute_transition_ringback_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [chan[3] := 255; partner[3] := 255; dev := 1] -> [true; chan[3] := 255; partner[3] := 255; dev := 1].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | chan[3] := 255.
                chan[3] = (255) & 0xff;
                // SLCO assignment | partner[3] := 255.
                partner[3] = (255) & 0xff;
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_3Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:1) | ringback -> calling | [record[3] != 255; partner[3] := record[3]].
            private boolean execute_transition_ringback_1() {
                // SLCO composite | [record[3] != 255; partner[3] := record[3]].
                // SLCO expression | record[3] != 255.
                if(!(record[3] != 255)) {
                    return false;
                }
                // SLCO assignment | partner[3] := record[3].
                partner[3] = (record[3]) & 0xff;

                currentState = GlobalClass_User_3Thread.States.calling;
                return true;
            }

            // SLCO transition (p:0, id:0) | talert -> errorstate | dev != 1 or chan[3] = 255.
            private boolean execute_transition_talert_0() {
                // SLCO expression | dev != 1 or chan[3] = 255.
                if(!(dev != 1 || chan[3] == 255)) {
                    return false;
                }

                currentState = GlobalClass_User_3Thread.States.errorstate;
                return true;
            }

            // SLCO transition (p:0, id:1) | talert -> tpickup | (chan[partner[3]] % 20) = 3.
            private boolean execute_transition_talert_1() {
                // SLCO expression | ((chan[partner[3]]) % 20) = 3 -> (chan[partner[3]] % 20) = 3.
                if(!((Math.floorMod(chan[partner[3]], 20)) == 3)) {
                    return false;
                }

                currentState = GlobalClass_User_3Thread.States.tpickup;
                return true;
            }

            // SLCO transition (p:0, id:2) | talert -> idle | (chan[partner[3]] % 20) != 3.
            private boolean execute_transition_talert_2() {
                // SLCO expression | ((chan[partner[3]]) % 20) != 3 -> (chan[partner[3]] % 20) != 3.
                if(!((Math.floorMod(chan[partner[3]], 20)) != 3)) {
                    return false;
                }

                currentState = GlobalClass_User_3Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | tpickup -> tconnected | [(chan[partner[3]] % 20) = 3 and (chan[partner[3]] / 20) = 0; dev := 0; chan[partner[3]] := (3 + 1 * 20); chan[3] := (partner[3] + 1 * 20)].
            private boolean execute_transition_tpickup_0() {
                // SLCO composite | [((chan[partner[3]]) % 20) = 3 and ((chan[partner[3]]) / 20) = 0; dev := 0; chan[partner[3]] := ((3) + (1) * 20); chan[3] := ((partner[3]) + (1) * 20)] -> [(chan[partner[3]] % 20) = 3 and (chan[partner[3]] / 20) = 0; dev := 0; chan[partner[3]] := (3 + 1 * 20); chan[3] := (partner[3] + 1 * 20)].
                // SLCO expression | (chan[partner[3]] % 20) = 3 and (chan[partner[3]] / 20) = 0.
                if(!((Math.floorMod(chan[partner[3]], 20)) == 3 && (chan[partner[3]] / 20) == 0)) {
                    return false;
                }
                // SLCO assignment | dev := 0.
                dev = (0) & 0xff;
                // SLCO assignment | chan[partner[3]] := (3 + 1 * 20).
                chan[partner[3]] = ((3 + 1 * 20)) & 0xff;
                // SLCO assignment | chan[3] := (partner[3] + 1 * 20).
                chan[3] = ((partner[3] + 1 * 20)) & 0xff;

                currentState = GlobalClass_User_3Thread.States.tconnected;
                return true;
            }

            // SLCO transition (p:0, id:1) | tpickup -> idle | [chan[partner[3]] = 255 or (chan[partner[3]] % 20) != 3; dev := 1; partner[3] := 255; chan[3] := 255].
            private boolean execute_transition_tpickup_1() {
                // SLCO composite | [chan[partner[3]] = 255 or ((chan[partner[3]]) % 20) != 3; dev := 1; partner[3] := 255; chan[3] := 255] -> [chan[partner[3]] = 255 or (chan[partner[3]] % 20) != 3; dev := 1; partner[3] := 255; chan[3] := 255].
                // SLCO expression | chan[partner[3]] = 255 or (chan[partner[3]] % 20) != 3.
                if(!(chan[partner[3]] == 255 || (Math.floorMod(chan[partner[3]], 20)) != 3)) {
                    return false;
                }
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;
                // SLCO assignment | partner[3] := 255.
                partner[3] = (255) & 0xff;
                // SLCO assignment | chan[3] := 255.
                chan[3] = (255) & 0xff;

                currentState = GlobalClass_User_3Thread.States.idle;
                return true;
            }

            // SLCO transition (p:0, id:0) | tconnected -> tconnected | [(chan[3] / 20) = 1 and dev = 0; dev := 1].
            private boolean execute_transition_tconnected_0() {
                // SLCO composite | [((chan[3]) / 20) = 1 and dev = 0; dev := 1] -> [(chan[3] / 20) = 1 and dev = 0; dev := 1].
                // SLCO expression | (chan[3] / 20) = 1 and dev = 0.
                if(!((chan[3] / 20) == 1 && dev == 0)) {
                    return false;
                }
                // SLCO assignment | dev := 1.
                dev = (1) & 0xff;

                currentState = GlobalClass_User_3Thread.States.tconnected;
                return true;
            }

            // SLCO transition (p:0, id:1) | tconnected -> tconnected | [(chan[3] / 20) = 1 and dev = 1; dev := 0].
            private boolean execute_transition_tconnected_1() {
                // SLCO composite | [((chan[3]) / 20) = 1 and dev = 1; dev := 0] -> [(chan[3] / 20) = 1 and dev = 1; dev := 0].
                // SLCO expression | (chan[3] / 20) = 1 and dev = 1.
                if(!((chan[3] / 20) == 1 && dev == 1)) {
                    return false;
                }
                // SLCO assignment | dev := 0.
                dev = (0) & 0xff;

                currentState = GlobalClass_User_3Thread.States.tconnected;
                return true;
            }

            // SLCO transition (p:0, id:2) | tconnected -> idle | [(chan[3] / 20) = 0; partner[3] := 255; chan[3] := 255].
            private boolean execute_transition_tconnected_2() {
                // SLCO composite | [((chan[3]) / 20) = 0; partner[3] := 255; chan[3] := 255] -> [(chan[3] / 20) = 0; partner[3] := 255; chan[3] := 255].
                // SLCO expression | (chan[3] / 20) = 0.
                if(!((chan[3] / 20) == 0)) {
                    return false;
                }
                // SLCO assignment | partner[3] := 255.
                partner[3] = (255) & 0xff;
                // SLCO assignment | chan[3] := 255.
                chan[3] = (255) & 0xff;

                currentState = GlobalClass_User_3Thread.States.idle;
                return true;
            }

            // Attempt to fire a transition starting in state idle.
            private void exec_idle() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | idle -> dialing | [chan[3] = 255; dev := 0; chan[3] := (3 + 0 * 20)].
                if(execute_transition_idle_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | idle -> qi | [chan[3] != 255; partner[3] := (chan[3] % 20)].
                if(execute_transition_idle_1()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state dialing.
            private void exec_dialing() {
                // [N_DET.START]
                switch(random.nextInt(6)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | dialing -> idle | true | [true; dev := 1; chan[3] := 255].
                        if(execute_transition_dialing_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (p:0, id:1) | dialing -> calling | true | partner[3] := 0.
                        if(execute_transition_dialing_1()) {
                            return;
                        }
                    }
                    case 2 -> {
                        // SLCO transition (p:0, id:2) | dialing -> calling | true | partner[3] := 1.
                        if(execute_transition_dialing_2()) {
                            return;
                        }
                    }
                    case 3 -> {
                        // SLCO transition (p:0, id:3) | dialing -> calling | true | partner[3] := 2.
                        if(execute_transition_dialing_3()) {
                            return;
                        }
                    }
                    case 4 -> {
                        // SLCO transition (p:0, id:4) | dialing -> calling | true | partner[3] := 3.
                        if(execute_transition_dialing_4()) {
                            return;
                        }
                    }
                    case 5 -> {
                        // SLCO transition (p:0, id:5) | dialing -> calling | true | partner[3] := 4.
                        if(execute_transition_dialing_5()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state calling.
            private void exec_calling() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | calling -> busy | partner[3] = 3.
                if(execute_transition_calling_0()) {
                    return;
                }
                // SLCO expression | partner[3] = 4.
                if(partner[3] == 4) {
                    // [N_DET.START]
                    switch(random.nextInt(2)) {
                        case 0 -> {
                            // SLCO transition (p:0, id:1) | calling -> unobtainable | partner[3] = 4.
                            if(execute_transition_calling_1()) {
                                return;
                            }
                        }
                        case 1 -> {
                            // SLCO transition (p:0, id:2) | calling -> ringback | partner[3] = 4.
                            if(execute_transition_calling_2()) {
                                return;
                            }
                        }
                    }
                    // [N_DET.END]
                }
                // SLCO transition (p:0, id:3) | calling -> busy | [partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255 and callforwardbusy[partner[3]] = 255; record[partner[3]] := 3].
                if(execute_transition_calling_3()) {
                    return;
                }
                // SLCO transition (p:0, id:4) | calling -> calling | [partner[3] != 3 and partner[3] != 4 and chan[partner[3]] != 255 and callforwardbusy[partner[3]] != 255; record[partner[3]] := 3; partner[3] := callforwardbusy[partner[3]]].
                if(execute_transition_calling_4()) {
                    return;
                }
                // SLCO transition (p:0, id:5) | calling -> oalert | [partner[3] != 3 and partner[3] != 4 and chan[partner[3]] = 255; record[partner[3]] := 3; chan[partner[3]] := (3 + 0 * 20); chan[3] := (partner[3] + 0 * 20)].
                if(execute_transition_calling_5()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state busy.
            private void exec_busy() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | busy -> idle | true | [true; chan[3] := 255; partner[3] := 255; dev := 1].
                if(execute_transition_busy_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state qi.
            private void exec_qi() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | qi -> talert | (chan[partner[3]] % 20) = 3.
                if(execute_transition_qi_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | qi -> idle | [(chan[partner[3]] % 20) != 3; partner[3] := 255].
                if(execute_transition_qi_1()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state talert.
            private void exec_talert() {
                // [N_DET.START]
                switch(random.nextInt(2)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | talert -> errorstate | dev != 1 or chan[3] = 255.
                        if(execute_transition_talert_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // [DET.START]
                        // SLCO transition (p:0, id:1) | talert -> tpickup | (chan[partner[3]] % 20) = 3.
                        if(execute_transition_talert_1()) {
                            return;
                        }
                        // SLCO transition (p:0, id:2) | talert -> idle | (chan[partner[3]] % 20) != 3.
                        if(execute_transition_talert_2()) {
                            return;
                        }
                        // [DET.END]
                    }
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state unobtainable.
            private void exec_unobtainable() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | unobtainable -> idle | true | [true; chan[3] := 255; partner[3] := 255; dev := 1].
                if(execute_transition_unobtainable_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state oalert.
            private void exec_oalert() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | oalert -> errorstate | (chan[3] % 20) != partner[3].
                if(execute_transition_oalert_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | oalert -> oconnected | (chan[3] % 20) = partner[3] and (chan[3] / 20) = 1.
                if(execute_transition_oalert_1()) {
                    return;
                }
                // SLCO transition (p:0, id:2) | oalert -> dveoringout | (chan[3] % 20) = partner[3] and (chan[3] / 20) = 0.
                if(execute_transition_oalert_2()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state errorstate.
            private void exec_errorstate() {
                // There are no transitions starting in state errorstate.
            }

            // Attempt to fire a transition starting in state oconnected.
            private void exec_oconnected() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | oconnected -> idle | true | [true; dev := 1; chan[3] := 255; chan[partner[3]] := 255].
                if(execute_transition_oconnected_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state dveoringout.
            private void exec_dveoringout() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | dveoringout -> idle | true | [true; dev := 1; chan[3] := 255; partner[3] := ((partner[3] % 20) + 0 * 20)].
                if(execute_transition_dveoringout_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state tpickup.
            private void exec_tpickup() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | tpickup -> tconnected | [(chan[partner[3]] % 20) = 3 and (chan[partner[3]] / 20) = 0; dev := 0; chan[partner[3]] := (3 + 1 * 20); chan[3] := (partner[3] + 1 * 20)].
                if(execute_transition_tpickup_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | tpickup -> idle | [chan[partner[3]] = 255 or (chan[partner[3]] % 20) != 3; dev := 1; partner[3] := 255; chan[3] := 255].
                if(execute_transition_tpickup_1()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state tconnected.
            private void exec_tconnected() {
                // [N_DET.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | tconnected -> tconnected | [(chan[3] / 20) = 1 and dev = 0; dev := 1].
                if(execute_transition_tconnected_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | tconnected -> tconnected | [(chan[3] / 20) = 1 and dev = 1; dev := 0].
                if(execute_transition_tconnected_1()) {
                    return;
                }
                // SLCO transition (p:0, id:2) | tconnected -> idle | [(chan[3] / 20) = 0; partner[3] := 255; chan[3] := 255].
                if(execute_transition_tconnected_2()) {
                    return;
                }
                // [DET.END]
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state ringback.
            private void exec_ringback() {
                // [N_DET.START]
                switch(random.nextInt(2)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | ringback -> idle | true | [true; chan[3] := 255; partner[3] := 255; dev := 1].
                        if(execute_transition_ringback_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (p:0, id:1) | ringback -> calling | [record[3] != 255; partner[3] := record[3]].
                        if(execute_transition_ringback_1()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 30) {
                    switch(currentState) {
                        case idle -> exec_idle();
                        case dialing -> exec_dialing();
                        case calling -> exec_calling();
                        case busy -> exec_busy();
                        case qi -> exec_qi();
                        case talert -> exec_talert();
                        case unobtainable -> exec_unobtainable();
                        case oalert -> exec_oalert();
                        case errorstate -> exec_errorstate();
                        case oconnected -> exec_oconnected();
                        case dveoringout -> exec_dveoringout();
                        case tpickup -> exec_tpickup();
                        case tconnected -> exec_tconnected();
                        case ringback -> exec_ringback();
                    }
                }
            }

            // The thread's run method.
            public void run() {
                try {
                    exec();
                } catch(Exception e) {
                    lockManager.exception_unlock();
                    throw e;
                }
            }
        }

        // Start all threads.
        public void startThreads() {
            T_User_0.start();
            T_User_1.start();
            T_User_2.start();
            T_User_3.start();
        }

        // Join all threads.
        public void joinThreads() {
            while (true) {
                try {
                    T_User_0.join();
                    T_User_1.join();
                    T_User_2.join();
                    T_User_3.join();
                    break;
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    // Start all threads.
    private void startThreads() {
        for(SLCO_Class object : objects) {
            object.startThreads();
        }
    }

    // Join all threads.
    private void joinThreads() {
        for(SLCO_Class object : objects) {
            object.joinThreads();
        }
    }

    // Run the application.
    public static void main(String[] args) {
        // Initialize the model and execute.
        Telephony model = new Telephony();
        model.startThreads();
        model.joinThreads();
    }
}