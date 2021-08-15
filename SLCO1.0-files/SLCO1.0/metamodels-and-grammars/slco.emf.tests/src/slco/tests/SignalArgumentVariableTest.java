/**
 */
package slco.tests;

import junit.textui.TestRunner;

import slco.SignalArgumentVariable;
import slco.SlcoFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Signal Argument Variable</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class SignalArgumentVariableTest extends SignalArgumentTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(SignalArgumentVariableTest.class);
	}

	/**
	 * Constructs a new Signal Argument Variable test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SignalArgumentVariableTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Signal Argument Variable test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected SignalArgumentVariable getFixture() {
		return (SignalArgumentVariable)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(SlcoFactory.eINSTANCE.createSignalArgumentVariable());
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#tearDown()
	 * @generated
	 */
	@Override
	protected void tearDown() throws Exception {
		setFixture(null);
	}

} //SignalArgumentVariableTest
