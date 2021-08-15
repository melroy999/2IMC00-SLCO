
import java.io.IOException;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.xtext.resource.XtextResourceSet;
import org.xtext.dve.DVEStandaloneSetup;
import com.google.inject.Injector;


public class DVE2XMI {
	
	
	public static void main(String[] args) {
		
		Injector injector = new DVEStandaloneSetup()
	            .createInjectorAndDoEMFRegistration();
	    XtextResourceSet resourceSet = injector
	            .getInstance(XtextResourceSet.class);
	    String absulotePath = "D:/GIT/SLCO/SLCO2.0/metamodels-and-grammars/dve2slco/models/peterson.1";
	    // .ext ist the extension of the model file
	    String inputURI = "file:///" + absulotePath + ".dve";
	    String outputURI = "file:///" + absulotePath + ".xmi";
	    URI uri = URI.createURI(inputURI);
	    Resource xtextResource = resourceSet.getResource(uri, true);

	    EcoreUtil.resolveAll(xtextResource);

	    Resource xmiResource = resourceSet
	            .createResource(URI.createURI(outputURI));
	    xmiResource.getContents().add(xtextResource.getContents().get(0));
	    try {
	        xmiResource.save(null);
	    } catch (IOException e) {
	        e.printStackTrace();
	    }
		
	    
	    //MyLanguageActivator.getInstance().getInjector(MyLanguageActivator.COM_MYCOMPANY_MYLANGUAGE).
		
	}
}