/**
 */
package objects.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import objects.ObjectLists;
import objects.ObjectsFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Object Lists</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class ObjectListsTest extends TestCase {

	/**
	 * The fixture for this Object Lists test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ObjectLists fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(ObjectListsTest.class);
	}

	/**
	 * Constructs a new Object Lists test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ObjectListsTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Object Lists test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(ObjectLists fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Object Lists test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ObjectLists getFixture() {
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
		setFixture(ObjectsFactory.eINSTANCE.createObjectLists());
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

} //ObjectListsTest
