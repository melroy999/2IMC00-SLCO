/**
 */
package poosl.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import poosl.PooslFactory;
import poosl.ProcessMethod;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Process Method</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class ProcessMethodTest extends TestCase {

	/**
	 * The fixture for this Process Method test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ProcessMethod fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(ProcessMethodTest.class);
	}

	/**
	 * Constructs a new Process Method test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessMethodTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Process Method test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(ProcessMethod fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Process Method test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ProcessMethod getFixture() {
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
		setFixture(PooslFactory.eINSTANCE.createProcessMethod());
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

} //ProcessMethodTest
