package GeneratedCode;

import java.util.ArrayList;
import java.util.Random;
import GenericCode.Class;
import GenericCode.SignalMessage;
import GenericCode.StateMachine;
import GenericCode.SharedVariable;
import GenericCode.SharedVariableList;


public class Rec1 extends StateMachine implements Runnable {
	protected boolean v;

    public Rec1(Class slcoClass) {
		super(slcoClass);
	    v = true;
	}
	
    public void run() {
		String currentState = "Rec1";
 		while(true){
			switch(currentState){
		 	case "Rec1":
                SharedVariableList s_Rec1toRec1 = new SharedVariableList();
			    SignalMessage signal = slcoClass.getPortName("port_In1").channel.receive(new ReceiveSignal_Rec1toRec1(slcoClass, s_Rec1toRec1, this),"Rec1toRec1", false);
				currentState = "Rec1";
				break;     
			}
		}
	}
} 
