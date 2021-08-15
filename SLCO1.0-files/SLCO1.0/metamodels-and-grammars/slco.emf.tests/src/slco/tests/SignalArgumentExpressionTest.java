/**
 */
package slco.tests;

import junit.textui.TestRunner;

import slco.SignalArgumentExpression;
import slco.SlcoFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Signal Argument Expression</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class SignalArgumentExpressionTest extends SignalArgumentTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(SignalArgumentExpressionTest.class);
	}

	/**
	 * Constructs a new Signal Argument Expression test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SignalArgumentExpressionTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Signal Argument Expression test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected SignalArgumentExpression getFixture() {
		return (SignalArgumentExpression)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(SlcoFactory.eINSTANCE.createSignalArgumentExpression());
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

} //SignalArgumentExpressionTest
