package GeneratedCode;

import java.util.ArrayList;
import java.util.Random;
import GenericCode.Class;
import GenericCode.SignalMessage;
import GenericCode.StateMachine;
import GenericCode.SharedVariable;
import GenericCode.SharedVariableList;


public class SendRec extends StateMachine implements Runnable {
	protected String s;

    public SendRec(Class slcoClass) {
		super(slcoClass);
	    s = "";
	}
	
    public void run() {
		String currentState = "SendRec0";
 		while(true){
			switch(currentState){
		 	case "SendRec0":
                SharedVariableList s_SendRec02SendRec1 = new SharedVariableList();
    
				s_SendRec02SendRec1.add(slcoClass.getSharedVariableName("m"));
				BooleanExpression_SendRec02SendRec1 expression_SendRec02SendRec1 = new BooleanExpression_SendRec02SendRec1(slcoClass, s_SendRec02SendRec1, this);
				expression_SendRec02SendRec1.lockVEvaluateUnlockV(false);
				currentState = "SendRec1";
				System.out.println("Transition: SendRec02SendRec1");
				break;     
		 	case "SendRec1":
				SharedVariableList s_SendRec12SendRec2 = new SharedVariableList();
				slcoClass.getPortName("port_InOut").channel.send(new SendSignal_SendRec12SendRec2(slcoClass, s_SendRec12SendRec2, this), "SendRec12SendRec2", false);
			    currentState = "SendRec2";
				break;     
		 	case "SendRec2":
                SharedVariableList s_SendRec22SendRec0 = new SharedVariableList();
			    SignalMessage signal = slcoClass.getPortName("port_InOut").channel.receive(new ReceiveSignal_SendRec22SendRec0(slcoClass, s_SendRec22SendRec0, this),"SendRec22SendRec0", false);
				currentState = "SendRec0";
				break;     
			}
		}
	}
} 
