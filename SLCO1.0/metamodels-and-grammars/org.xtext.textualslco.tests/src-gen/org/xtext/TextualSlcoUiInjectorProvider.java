/*
* generated by Xtext
*/
package org.xtext;

import org.eclipse.xtext.junit4.IInjectorProvider;

import com.google.inject.Injector;

public class TextualSlcoUiInjectorProvider implements IInjectorProvider {
	
	public Injector getInjector() {
		return org.xtext.ui.internal.TextualSlcoActivator.getInstance().getInjector("org.xtext.TextualSlco");
	}
	
}
