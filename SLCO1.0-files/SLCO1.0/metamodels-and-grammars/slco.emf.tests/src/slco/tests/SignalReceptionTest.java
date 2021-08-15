/**
 */
package slco.tests;

import junit.textui.TestRunner;

import slco.SignalReception;
import slco.SlcoFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Signal Reception</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class SignalReceptionTest extends StatementTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(SignalReceptionTest.class);
	}

	/**
	 * Constructs a new Signal Reception test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SignalReceptionTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Signal Reception test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected SignalReception getFixture() {
		return (SignalReception)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(SlcoFactory.eINSTANCE.createSignalReception());
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

} //SignalReceptionTest
