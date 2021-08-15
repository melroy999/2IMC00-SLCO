
package org.xtext;

/**
 * Initialization support for running Xtext languages 
 * without equinox extension registry
 */
public class TextualSlcoStandaloneSetup extends TextualSlcoStandaloneSetupGenerated{

	public static void doSetup() {
		new TextualSlcoStandaloneSetup().createInjectorAndDoEMFRegistration();
	}
}

