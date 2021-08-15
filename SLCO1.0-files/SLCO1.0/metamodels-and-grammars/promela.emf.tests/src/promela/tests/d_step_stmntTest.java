/**
 */
package promela.tests;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.d_step_stmnt;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>dstep stmnt</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class d_step_stmntTest extends stmntTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(d_step_stmntTest.class);
	}

	/**
	 * Constructs a new dstep stmnt test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public d_step_stmntTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this dstep stmnt test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected d_step_stmnt getFixture() {
		return (d_step_stmnt)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.created_step_stmnt());
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

} //d_step_stmntTest
