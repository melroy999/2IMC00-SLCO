/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.AssignmentStatement;
import nqc.NqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Assignment Statement</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class AssignmentStatementTest extends StatementTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(AssignmentStatementTest.class);
	}

	/**
	 * Constructs a new Assignment Statement test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public AssignmentStatementTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Assignment Statement test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected AssignmentStatement getFixture() {
		return (AssignmentStatement)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createAssignmentStatement());
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

} //AssignmentStatementTest
