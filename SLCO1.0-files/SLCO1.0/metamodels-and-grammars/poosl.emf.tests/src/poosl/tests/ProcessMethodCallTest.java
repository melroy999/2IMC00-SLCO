/**
 */
package poosl.tests;

import junit.textui.TestRunner;

import poosl.PooslFactory;
import poosl.ProcessMethodCall;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Process Method Call</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class ProcessMethodCallTest extends StatementTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(ProcessMethodCallTest.class);
	}

	/**
	 * Constructs a new Process Method Call test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessMethodCallTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Process Method Call test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected ProcessMethodCall getFixture() {
		return (ProcessMethodCall)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PooslFactory.eINSTANCE.createProcessMethodCall());
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

} //ProcessMethodCallTest
