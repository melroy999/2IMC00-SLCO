package GeneratedCode;

import java.util.ArrayList;
import java.util.Random;
import GenericCode.Class;
import GenericCode.SignalMessage;
import GenericCode.StateMachine;
import GenericCode.SharedVariable;
import GenericCode.SharedVariableList;


public class Com extends StateMachine implements Runnable {
	protected String s;

    public Com(Class slcoClass) {
		super(slcoClass);
	    s = "";
	}
	
    public void run() {
		String currentState = "Com0";
 		while(true){
			switch(currentState){
		 	case "Com0":
				long start = System.currentTimeMillis();
				boolean isExecutableTransition = false;
				String nextTransition; 
				String[] transitions = {"Com02Com1","Com02Com2"};
				while(!isExecutableTransition){
					int idx = new Random().nextInt(transitions.length);
 					nextTransition = transitions[idx];
					while(nextTransition.equals("Com02Com2")){
						if(nextTransition.equals("Com02Com2")){
							if((System.currentTimeMillis() - start) >= 5){
								break;
							}
						}
						idx = new Random().nextInt(transitions.length);
						nextTransition = transitions[idx];
					}
					switch(nextTransition){
					case "Com02Com1":
                			SharedVariableList s_Com02Com1 = new SharedVariableList();
                            boolean isExecutable_Com02Com1 = false;
							isExecutable_Com02Com1 =slcoClass.getPortName("port_Out1").channel.send(new SendSignal_Com02Com1(slcoClass, s_Com02Com1, this), "Com02Com1", true);
							if(isExecutable_Com02Com1){
								isExecutableTransition = true;
								currentState = "Com1";
							}
							break;                 
					case "Com02Com2":
							isExecutableTransition = true;
							currentState = "Com2";
							System.out.println("Transition: Com02Com2");
							break;                 
					}
				}        
				break;
		 	case "Com1":
				SharedVariableList s_Com12Com3 = new SharedVariableList();
				slcoClass.getPortName("port_Out2").channel.send(new SendSignal_Com12Com3(slcoClass, s_Com12Com3, this), "Com12Com3", false);
			    currentState = "Com3";
				break;     
		 	case "Com3":
                SharedVariableList s_Com32Com4 = new SharedVariableList();
			    SignalMessage signal = slcoClass.getPortName("port_InOut").channel.receive(new ReceiveSignal_Com32Com4(slcoClass, s_Com32Com4, this),"Com32Com4", false);
				currentState = "Com4";
				break;     
		 	case "Com4":
				SharedVariableList s_Com42Com2 = new SharedVariableList();
				slcoClass.getPortName("port_InOut").channel.send(new SendSignal_Com42Com2(slcoClass, s_Com42Com2, this), "Com42Com2", false);
			    currentState = "Com2";
				break;     
			case "Com2":
				return;
			}
		}
	}
} 
