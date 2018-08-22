
import java.io.IOException;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.xtext.resource.XtextResourceSet;
import org.xtext.slco.textualslco.TextualSlcoStandaloneSetup;
import com.google.inject.Injector;


public class XMI2SLCO {
	
	
	public static void main(String[] args) {
		
		Injector injector = new TextualSlcoStandaloneSetup()
	            .createInjectorAndDoEMFRegistration();
	    XtextResourceSet resourceSet = injector
	            .getInstance(XtextResourceSet.class);
	    String absulotePath = "D:/GIT/SLCO/SLCOtoSLCO_Verification/eclipse_workspace/Example1/example_init_transformed_transformed";
	    // .ext ist the extension of the model file
	    String inputURI = "file:///" + absulotePath + ".xmi";
	    String outputURI = "file:///" + absulotePath + ".slcotxt";
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