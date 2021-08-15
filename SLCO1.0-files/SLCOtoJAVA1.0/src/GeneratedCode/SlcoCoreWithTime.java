package GeneratedCode;

import GenericCode.AsynchronousChannel;
import GenericCode.Channel;
import GenericCode.Model;
import GenericCode.SynchronousChannel;
	
public class SlcoCoreWithTime extends Model{
	Channel c1;
	Channel c2;
	Channel c3;
	
	public SlcoCoreWithTime(String name) {
		super(name);
	    c1 = new AsynchronousChannel();
	    channels.put("c1", c1);
	    c2 = new AsynchronousChannel();
	    channels.put("c2", c2);
		c3 = new SynchronousChannel();
		channels.put("c3", c3);
    	slcoObjects.put("q", new Q("Q", "q", this, c2, c1, c3));
    	
    	slcoObjects.put("p", new P("P", "p", this, c2, c1, c3));
    	
	}
}