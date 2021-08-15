package GeneratedCode;

import java.util.ArrayList;
import GenericCode.SharedVariable;
import GenericCode.BooleanExpression;
import GenericCode.Class;
import GenericCode.SignalMessage;
import GenericCode.ReceiveSignal;
import GenericCode.SharedVariableList;


public class ReceiveSignal_SendRec22SendRec0 extends ReceiveSignal{
	public SendRec sm;
	public ReceiveSignal_SendRec22SendRec0(Class slcoClass, SharedVariableList sharedVariables, SendRec sm) {
		super(slcoClass, sharedVariables);
		this.sm = sm;
		// TODO Auto-generated constructor stub
	}
	@Override
	public boolean evaluate() {
		// TODO Auto-generated method stub
		SignalMessage signal = slcoClass.getPortName("port_InOut").channel.peek();
		return ((String)signal.name).equals("T") && signal.args.length == 1 && true;
	}	
    @Override
	public void receive() {
		// TODO Auto-generated method stub
		SignalMessage signal = slcoClass.getPortName("port_InOut").channel.peek();
		sm.s=(String)signal.args[0];
	}	
}
