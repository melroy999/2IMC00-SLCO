/**
 */
package poosl.tests;

import junit.textui.TestRunner;

import poosl.Abort;
import poosl.PooslFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Abort</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class AbortTest extends StatementTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(AbortTest.class);
	}

	/**
	 * Constructs a new Abort test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public AbortTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Abort test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected Abort getFixture() {
		return (Abort)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PooslFactory.eINSTANCE.createAbort());
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

} //AbortTest
