/**
 */
package poosl.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import poosl.PooslFactory;
import poosl.ProcessInstance;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Process Instance</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class ProcessInstanceTest extends TestCase {

	/**
	 * The fixture for this Process Instance test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ProcessInstance fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(ProcessInstanceTest.class);
	}

	/**
	 * Constructs a new Process Instance test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessInstanceTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Process Instance test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(ProcessInstance fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Process Instance test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ProcessInstance getFixture() {
		return fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PooslFactory.eINSTANCE.createProcessInstance());
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

} //ProcessInstanceTest
