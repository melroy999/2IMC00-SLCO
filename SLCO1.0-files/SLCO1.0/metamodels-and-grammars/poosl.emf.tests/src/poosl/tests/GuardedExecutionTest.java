/**
 */
package poosl.tests;

import junit.textui.TestRunner;

import poosl.GuardedExecution;
import poosl.PooslFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Guarded Execution</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class GuardedExecutionTest extends StatementTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(GuardedExecutionTest.class);
	}

	/**
	 * Constructs a new Guarded Execution test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public GuardedExecutionTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Guarded Execution test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected GuardedExecution getFixture() {
		return (GuardedExecution)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PooslFactory.eINSTANCE.createGuardedExecution());
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

} //GuardedExecutionTest
