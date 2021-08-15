package GenericCode;

public class SignalMessage {
	public String name;
	public Object[] args;
	public SignalMessage(String name){
		this.name = name;
		this.args = null;
	}
	public SignalMessage(String name, Object...args){
		this.name = name;
		this.args = args;
	}
}
