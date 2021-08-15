package GenericCode;
public class Port {
	public Channel channel;
	private String name;
	public Port(Channel channel){
		this.channel = channel;
	}
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
}
