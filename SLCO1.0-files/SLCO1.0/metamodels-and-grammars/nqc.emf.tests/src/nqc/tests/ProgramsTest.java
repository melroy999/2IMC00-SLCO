/**
 */
package nqc.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.Programs;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Programs</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class ProgramsTest extends TestCase {

	/**
	 * The fixture for this Programs test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Programs fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(ProgramsTest.class);
	}

	/**
	 * Constructs a new Programs test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProgramsTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Programs test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(Programs fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Programs test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Programs getFixture() {
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
		setFixture(NqcFactory.eINSTANCE.createPrograms());
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

} //ProgramsTest
