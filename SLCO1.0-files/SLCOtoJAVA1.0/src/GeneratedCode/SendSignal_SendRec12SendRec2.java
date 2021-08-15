package GeneratedCode;

import java.util.ArrayList;
import GenericCode.SharedVariable;
import GenericCode.Class;
import GenericCode.SignalMessage;
import GenericCode.SendSignal;
import GenericCode.SharedVariableList;

public class SendSignal_SendRec12SendRec2 extends SendSignal{
	public SendRec sm;
	public SendSignal_SendRec12SendRec2(Class slcoClass, SharedVariableList sharedVariables, SendRec sm) {
		super(slcoClass, sharedVariables);
		this.sm = sm;
		// TODO Auto-generated constructor stub
	}
	@Override
	public boolean send(boolean isNonDeterministicTransition) {
		// TODO Auto-generated method stub
  
        SignalMessage signal = new SignalMessage("S",new Object[]{"a"});
        return slcoClass.getPortName("port_InOut").channel.put(signal, isNonDeterministicTransition);
	}
}
