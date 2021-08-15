package GeneratedCode;

import java.util.ArrayList;
import GenericCode.SharedVariable;
import GenericCode.Class;
import GenericCode.SignalMessage;
import GenericCode.SendSignal;
import GenericCode.SharedVariableList;

public class SendSignal_Com42Com2 extends SendSignal{
	public Com sm;
	public SendSignal_Com42Com2(Class slcoClass, SharedVariableList sharedVariables, Com sm) {
		super(slcoClass, sharedVariables);
		this.sm = sm;
		// TODO Auto-generated constructor stub
	}
	@Override
	public boolean send(boolean isNonDeterministicTransition) {
		// TODO Auto-generated method stub
  
        SignalMessage signal = new SignalMessage("T",new Object[]{sm.s});
        return slcoClass.getPortName("port_InOut").channel.put(signal, isNonDeterministicTransition);
	}
}
