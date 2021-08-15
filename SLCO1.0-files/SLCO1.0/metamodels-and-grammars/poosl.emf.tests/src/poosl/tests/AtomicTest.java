/**
 */
package poosl.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import poosl.Atomic;
import poosl.PooslFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Atomic</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class AtomicTest extends TestCase {

	/**
	 * The fixture for this Atomic test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Atomic fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(AtomicTest.class);
	}

	/**
	 * Constructs a new Atomic test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public AtomicTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Atomic test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(Atomic fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Atomic test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Atomic getFixture() {
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
		setFixture(PooslFactory.eINSTANCE.createAtomic());
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

} //AtomicTest
