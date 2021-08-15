/**
 */
package promela.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.mtype;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>mtype</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class mtypeTest extends TestCase {

	/**
	 * The fixture for this mtype test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected mtype fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(mtypeTest.class);
	}

	/**
	 * Constructs a new mtype test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public mtypeTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this mtype test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(mtype fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this mtype test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected mtype getFixture() {
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
		setFixture(PromelaFactory.eINSTANCE.createmtype());
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

} //mtypeTest
