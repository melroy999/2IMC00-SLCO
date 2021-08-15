package GenericCode;

import java.util.HashMap;
import java.util.Map;
//public class SLCOModel {

public class Model {
	
	private String SLCOModelName = null;
	protected Map<String, Class> slcoObjects;
	protected Map<String, Channel> channels;

	public Model(String name) {
		
		setSLCOModelName(name);
		slcoObjects = new HashMap<String, Class>();
		channels = new HashMap<String, Channel>();
		
	}

	public Map<String, Class> getSlcoObjects() {
		return slcoObjects;
	}

	public void setSlcoObjects(Map<String, Class> slcoObjects) {
		this.slcoObjects = slcoObjects;
	}

	public Map<String, Channel> getChannels() {
		return channels;
	}

	public void setChannels(Map<String, Channel> channels) {
		this.channels = channels;
	}

	public String getSLCOModelName() {
		return SLCOModelName;
	}

	public void setSLCOModelName(String sLCOModelName) {
		SLCOModelName = sLCOModelName;
	}
}
