package GeneratedCode;

import java.util.ArrayList;
import java.util.Random;
import GenericCode.Class;
import GenericCode.SignalMessage;
import GenericCode.StateMachine;
import GenericCode.SharedVariable;
import GenericCode.SharedVariableList;


public class Rec2 extends StateMachine implements Runnable {

    public Rec2(Class slcoClass) {
		super(slcoClass);
	}
	
    public void run() {
		String currentState = "Rec2a";
 		while(true){
			switch(currentState){
		 	case "Rec2a":
                SharedVariableList s_Rec2a2Rec2b = new SharedVariableList();
    
				s_Rec2a2Rec2b.add(slcoClass.getSharedVariableName("m"));
			    SignalMessage signal = slcoClass.getPortName("port_In2").channel.receive(new ReceiveSignal_Rec2a2Rec2b(slcoClass, s_Rec2a2Rec2b, this),"Rec2a2Rec2b", false);
				currentState = "Rec2b";
				break;     
		 	case "Rec2b":
						
				SharedVariableList s_Rec2b2Rec2a = new SharedVariableList();
    
				s_Rec2b2Rec2a.add(slcoClass.getSharedVariableName("m"));
				
				Assignment_Rec2b2Rec2a assignment_Rec2b2Rec2a = new Assignment_Rec2b2Rec2a(slcoClass, s_Rec2b2Rec2a, this);
				assignment_Rec2b2Rec2a.lockVAssignUnlockV();
				currentState = "Rec2a";
				System.out.println("Transition: Rec2b2Rec2a");
				break;     
			}
		}
	}
} 
