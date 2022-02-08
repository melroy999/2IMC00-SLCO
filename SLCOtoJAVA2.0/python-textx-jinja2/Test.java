// main class
public class Test {
  // Transition functions

  /*@
  pure boolean value_SMC0_0_x(int x_index, int i3, boolean b2, int i1, boolean b1, boolean x_old) {
  	return ((x_index == i3) ? b2 : ((x_index == i1) ? b1 : x_old));
  }

  given boolean[] x;
  given int y;
  given int i;
  given int j;
  given boolean b0;
  given int i0;
  given int i1;
  given boolean b1;
  given int i2;
  given int i3;
  given boolean b2;
  invariant x != null;
  context (\forall* int slco_i ; 0 <= slco_i < x.length ; Perm(x[slco_i],write));
  ensures b0 ==> \result == true;
  ensures !b0 ==> \result == false;
  ensures b0 ==> (\forall* int slco_i ; 0 <= slco_i < x.length ; x[slco_i] == value_SMC0_0_x(slco_i, i3, b2, i1, b1, \old(x[slco_i])));
  ensures !b0 ==> (\forall* int slco_i ; 0 <= slco_i < x.length ; x[slco_i] == \old(x[slco_i]));
  ensures b0 ==> (i == i2);
  ensures !b0 ==> (i == \old(i));
  @*/
  boolean execute_SMC0_0() {
    // SLCO statement: [ not x[i]; i := i + 1; x[i] := i = 2; i := 3; x[0] := False ]
    /*@ assume 0 <= i < x.length; @*/
    /*@ b0 = !(x[i]); @*/
    if (!(!(x[i]))) { return false; }
    /*@ i0 = i + 1; @*/
    i = i + 1;
    /*@ assume 0 <= i < x.length; @*/
    /*@ i1 = i; b1 = i == 2; @*/
    x[i] = i == 2;
    /*@ i2 = 3; @*/
    i = 3;
    /*@ assume 0 <= 0 < x.length; @*/
    /*@ i3 = 0; b2 = false; @*/
    x[0] = false;
    return true;
  }

  /*@
  pure boolean value_SMC0_1_x(int x_index, int i2, boolean b2, int i0, boolean b1, boolean x_old) {
  	return ((x_index == i2) ? b2 : ((x_index == i0) ? b1 : x_old));
  }

  given boolean[] x;
  given int y;
  given int i;
  given int j;
  given boolean b0;
  given int i0;
  given boolean b1;
  given int i1;
  given int i2;
  given boolean b2;
  invariant x != null;
  context (\forall* int slco_i ; 0 <= slco_i < x.length ; Perm(x[slco_i],write));
  ensures b0 ==> \result == true;
  ensures !b0 ==> \result == false;
  ensures b0 ==> (j == i1);
  ensures !b0 ==> (j == \old(j));
  ensures b0 ==> (\forall* int slco_i ; 0 <= slco_i < x.length ; x[slco_i] == value_SMC0_1_x(slco_i, i2, b2, i0, b1, \old(x[slco_i])));
  ensures !b0 ==> (\forall* int slco_i ; 0 <= slco_i < x.length ; x[slco_i] == \old(x[slco_i]));
  @*/
  boolean execute_SMC0_1() {
    // SLCO statement: [ not x[j + 0]; x[j + 1] := j = 2; j := 3; x[0] := False ]
    /*@ assume 0 <= j + 0 < x.length; @*/
    /*@ b0 = !(x[j + 0]); @*/
    if (!(!(x[j + 0]))) { return false; }
    /*@ assume 0 <= j + 1 < x.length; @*/
    /*@ i0 = j + 1; b1 = j == 2; @*/
    x[j + 1] = j == 2;
    /*@ i1 = 3; @*/
    j = 3;
    /*@ assume 0 <= 0 < x.length; @*/
    /*@ i2 = 0; b2 = false; @*/
    x[0] = false;
    return true;
  }

  /*@
  given boolean[] x;
  given int y;
  given int i;
  given int j;
  invariant x != null;
  ensures \result == true;
  ensures (i == 0);
  @*/
  boolean execute_SMC0_2() {
    // SLCO statement: i := 0
    i = 0;
    return true;
  }

  /*@
  given boolean[] x;
  given int y;
  given int lx;
  invariant x != null;
  ensures (lx == 0) ==> \result == true;
  ensures !(lx == 0) ==> \result == false;
  @*/
  boolean execute_Com0_0() {
    // SLCO statement: lx = 0
    if (!(lx == 0)) { return false; }
    return true;
  }

  /*@
  given boolean[] x;
  given int y;
  @*/
  // Constructor for main class
  Test() {
    // Instantiate global variables
    x = new boolean[] {false,true};
    y = 0;
  }
}