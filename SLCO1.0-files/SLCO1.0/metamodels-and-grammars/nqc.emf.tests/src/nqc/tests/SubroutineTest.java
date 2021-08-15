/**
 */
package nqc.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.Subroutine;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Subroutine</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class SubroutineTest extends TestCase {

	/**
	 * The fixture for this Subroutine test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Subroutine fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(SubroutineTest.class);
	}

	/**
	 * Constructs a new Subroutine test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SubroutineTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Subroutine test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(Subroutine fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Subroutine test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Subroutine getFixture() {
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
		setFixture(NqcFactory.eINSTANCE.createSubroutine());
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

} //SubroutineTest
