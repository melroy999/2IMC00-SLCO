package GenericCode;

import java.util.ArrayList;

public final class SharedVariableList {
	protected ArrayList <SharedVariable> elements;
	
	public SharedVariableList(){
		elements = new ArrayList<SharedVariable>();
	}
	
	public int size(){
		return elements.size();
	}
	
    public SharedVariable get(int index){
    	return elements.get(index);
    }
    public boolean add(SharedVariable v){
    	return elements.add(v);
    }
}
